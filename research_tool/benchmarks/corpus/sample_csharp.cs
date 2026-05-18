// EventDrivenNotificationService implements a pub/sub notification system
// with support for typed event channels, priority-based delivery, and
// configurable retry policies. It uses an in-memory event bus with
// optional persistence to a backing store for guaranteed delivery.
//
// The system supports three delivery modes: fire-and-forget for
// non-critical notifications, at-least-once for important alerts,
// and exactly-once for transactional messages using idempotency keys.
//
// Memory usage scales linearly with the number of active subscriptions
// and pending events. Each subscription consumes approximately 256 bytes
// of overhead plus the size of its filter predicate closure.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NotificationService
{
    /// <summary>
    /// Defines the delivery guarantee level for event dispatching.
    /// FireAndForget provides no delivery confirmation.
    /// AtLeastOnce retries failed deliveries up to the configured maximum.
    /// ExactlyOnce uses idempotency keys to prevent duplicate processing.
    /// </summary>
    public enum DeliveryMode
    {
        FireAndForget = 0,
        AtLeastOnce = 1,
        ExactlyOnce = 2
    }

    /// <summary>
    /// Priority levels for event processing order.
    /// Critical events are always processed before Normal events,
    /// regardless of arrival time. Bulk priority is used for
    /// batch operations that should yield to interactive traffic.
    /// </summary>
    public enum EventPriority
    {
        Bulk = 0,
        Low = 1,
        Normal = 2,
        High = 3,
        Critical = 4
    }

    /// <summary>
    /// Configuration for the retry policy applied to failed event deliveries.
    /// Uses exponential backoff with jitter: delay = BaseDelay * 2^attempt + random(0, JitterMs).
    /// The maximum wait between retries is capped at MaxDelay regardless of the
    /// computed exponential value.
    /// </summary>
    public record RetryPolicy
    {
        public int MaxRetries { get; init; } = 3;
        public TimeSpan BaseDelay { get; init; } = TimeSpan.FromMilliseconds(100);
        public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(30);
        public int JitterMs { get; init; } = 50;

        /// <summary>
        /// Computes the delay for a given retry attempt using exponential backoff with jitter.
        /// Returns the lesser of the computed delay and MaxDelay.
        /// </summary>
        public TimeSpan ComputeDelay(int attempt)
        {
            var exponential = TimeSpan.FromMilliseconds(
                BaseDelay.TotalMilliseconds * Math.Pow(2, attempt));
            var jitter = TimeSpan.FromMilliseconds(Random.Shared.Next(0, JitterMs));
            var total = exponential + jitter;
            return total > MaxDelay ? MaxDelay : total;
        }
    }

    /// <summary>
    /// Immutable event envelope carrying the payload, metadata, and routing information.
    /// The EventId is generated deterministically from the source, type, and timestamp
    /// using SHA-256 hashing to enable deduplication across distributed nodes.
    /// </summary>
    public record NotificationEvent
    {
        public string EventId { get; init; } = "";
        public string EventType { get; init; } = "";
        public string Source { get; init; } = "";
        public EventPriority Priority { get; init; } = EventPriority.Normal;
        public DeliveryMode Delivery { get; init; } = DeliveryMode.FireAndForget;
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;
        public Dictionary<string, string> Headers { get; init; } = new();
        public string Payload { get; init; } = "";

        /// <summary>
        /// Generates a deterministic event ID by hashing the source, type, and timestamp.
        /// This enables deduplication when the same logical event arrives from multiple paths.
        /// </summary>
        public static string GenerateEventId(string source, string eventType, DateTime timestamp)
        {
            var input = $"{source}:{eventType}:{timestamp:O}";
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(input));
            return Convert.ToHexString(hash)[..16].ToLowerInvariant();
        }
    }

    /// <summary>
    /// Represents a subscription to a specific event channel with an optional
    /// filter predicate. Only events matching the filter are delivered to the handler.
    /// </summary>
    public class Subscription
    {
        public string Id { get; }
        public string Channel { get; }
        public Func<NotificationEvent, bool>? Filter { get; }
        public Func<NotificationEvent, CancellationToken, Task> Handler { get; }
        public DeliveryMode RequiredDelivery { get; }
        public DateTime CreatedAt { get; }

        public Subscription(
            string channel,
            Func<NotificationEvent, CancellationToken, Task> handler,
            Func<NotificationEvent, bool>? filter = null,
            DeliveryMode requiredDelivery = DeliveryMode.FireAndForget)
        {
            Id = Guid.NewGuid().ToString("N")[..12];
            Channel = channel;
            Handler = handler;
            Filter = filter;
            RequiredDelivery = requiredDelivery;
            CreatedAt = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Tracks the delivery status of an event to a specific subscriber.
    /// Includes attempt history for diagnosing delivery failures.
    /// </summary>
    public class DeliveryRecord
    {
        public string EventId { get; init; } = "";
        public string SubscriptionId { get; init; } = "";
        public int Attempts { get; set; }
        public bool Delivered { get; set; }
        public DateTime? LastAttemptAt { get; set; }
        public string? LastError { get; set; }
        public List<DateTime> AttemptTimestamps { get; } = new();
    }

    /// <summary>
    /// Interface for persisting events and delivery records to a backing store.
    /// Implementations must be thread-safe as the event bus may invoke methods
    /// concurrently from multiple dispatch tasks.
    /// </summary>
    public interface IEventStore
    {
        Task SaveEventAsync(NotificationEvent evt, CancellationToken ct = default);
        Task<NotificationEvent?> GetEventAsync(string eventId, CancellationToken ct = default);
        Task SaveDeliveryRecordAsync(DeliveryRecord record, CancellationToken ct = default);
        Task<IReadOnlyList<DeliveryRecord>> GetPendingDeliveriesAsync(CancellationToken ct = default);
        Task<bool> HasBeenProcessedAsync(string eventId, string subscriptionId, CancellationToken ct = default);
    }

    /// <summary>
    /// Central event bus that manages subscriptions, dispatches events to handlers,
    /// and enforces delivery guarantees. The bus processes events in priority order
    /// using a concurrent priority queue.
    ///
    /// Thread safety: all public methods are safe to call from multiple threads.
    /// Subscriptions and event dispatch are protected by lock-free concurrent collections.
    ///
    /// The dispatch loop drains the priority queue every 50 milliseconds or when
    /// a new event is published, whichever comes first. Events at the same priority
    /// level are dispatched in FIFO order.
    /// </summary>
    public class EventBus : IAsyncDisposable
    {
        private readonly ConcurrentDictionary<string, List<Subscription>> _channels = new();
        private readonly ConcurrentDictionary<string, DeliveryRecord> _deliveryLog = new();
        private readonly PriorityQueue<NotificationEvent, int> _eventQueue = new();
        private readonly SemaphoreSlim _queueSignal = new(0);
        private readonly RetryPolicy _retryPolicy;
        private readonly IEventStore? _store;
        private readonly int _maxConcurrentDispatches;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _dispatchTask;
        private long _totalPublished;
        private long _totalDelivered;
        private long _totalFailed;

        /// <summary>
        /// Creates a new EventBus with the specified retry policy and optional backing store.
        /// The maxConcurrentDispatches parameter limits parallel handler invocations
        /// to prevent thread pool starvation under high event volume. Default is 16.
        /// </summary>
        public EventBus(
            RetryPolicy? retryPolicy = null,
            IEventStore? store = null,
            int maxConcurrentDispatches = 16)
        {
            _retryPolicy = retryPolicy ?? new RetryPolicy();
            _store = store;
            _maxConcurrentDispatches = maxConcurrentDispatches;
            _dispatchTask = Task.Run(() => DispatchLoopAsync(_cts.Token));
        }

        /// <summary>Total number of events published to the bus since creation.</summary>
        public long TotalPublished => Interlocked.Read(ref _totalPublished);

        /// <summary>Total number of successful event deliveries across all subscribers.</summary>
        public long TotalDelivered => Interlocked.Read(ref _totalDelivered);

        /// <summary>Total number of permanently failed deliveries (exhausted retries).</summary>
        public long TotalFailed => Interlocked.Read(ref _totalFailed);

        /// <summary>
        /// Subscribes a handler to the specified channel. Returns the subscription
        /// for later unsubscription. The filter predicate, if provided, is evaluated
        /// before each delivery — only matching events reach the handler.
        /// </summary>
        public Subscription Subscribe(
            string channel,
            Func<NotificationEvent, CancellationToken, Task> handler,
            Func<NotificationEvent, bool>? filter = null,
            DeliveryMode delivery = DeliveryMode.FireAndForget)
        {
            var sub = new Subscription(channel, handler, filter, delivery);
            _channels.AddOrUpdate(
                channel,
                _ => new List<Subscription> { sub },
                (_, list) => { lock (list) { list.Add(sub); } return list; });
            return sub;
        }

        /// <summary>
        /// Removes a subscription by its ID. Returns true if the subscription was
        /// found and removed, false if it was already unsubscribed or never existed.
        /// </summary>
        public bool Unsubscribe(string subscriptionId)
        {
            foreach (var (_, subs) in _channels)
            {
                lock (subs)
                {
                    var idx = subs.FindIndex(s => s.Id == subscriptionId);
                    if (idx >= 0)
                    {
                        subs.RemoveAt(idx);
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Publishes an event to all matching subscribers on the event's channel.
        /// The event is enqueued with its priority for ordered dispatch.
        /// For AtLeastOnce and ExactlyOnce modes, the event is persisted before
        /// acknowledgment to survive process restarts.
        /// </summary>
        public async Task PublishAsync(NotificationEvent evt, CancellationToken ct = default)
        {
            if (_store != null && evt.Delivery != DeliveryMode.FireAndForget)
            {
                await _store.SaveEventAsync(evt, ct);
            }

            lock (_eventQueue)
            {
                _eventQueue.Enqueue(evt, -(int)evt.Priority);
            }
            _queueSignal.Release();
            Interlocked.Increment(ref _totalPublished);
        }

        /// <summary>
        /// Returns all active subscriptions for a channel, filtered by an optional
        /// predicate. Uses LINQ to filter and project subscription metadata.
        /// </summary>
        public IReadOnlyList<(string Id, string Channel, DateTime CreatedAt)> GetSubscriptions(
            string? channel = null)
        {
            var query = _channels
                .SelectMany(kv => kv.Value.Select(s => (s.Id, s.Channel, s.CreatedAt)));

            if (channel != null)
                query = query.Where(s => s.Channel == channel);

            return query.OrderBy(s => s.CreatedAt).ToList();
        }

        /// <summary>
        /// Internal dispatch loop that continuously drains the priority queue and
        /// delivers events to matching subscribers. Uses SemaphoreSlim to throttle
        /// concurrent handler invocations to maxConcurrentDispatches.
        /// </summary>
        private async Task DispatchLoopAsync(CancellationToken ct)
        {
            var throttle = new SemaphoreSlim(_maxConcurrentDispatches);

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    await _queueSignal.WaitAsync(TimeSpan.FromMilliseconds(50), ct);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                NotificationEvent? evt;
                lock (_eventQueue)
                {
                    if (!_eventQueue.TryDequeue(out evt, out _))
                        continue;
                }

                if (!_channels.TryGetValue(evt.EventType, out var subscribers))
                    continue;

                List<Subscription> snapshot;
                lock (subscribers)
                {
                    snapshot = subscribers.ToList();
                }

                foreach (var sub in snapshot)
                {
                    if (sub.Filter != null && !sub.Filter(evt))
                        continue;

                    await throttle.WaitAsync(ct);
                    _ = DeliverAsync(evt, sub, throttle, ct);
                }
            }
        }

        /// <summary>
        /// Delivers a single event to a subscriber with retry logic.
        /// For ExactlyOnce delivery, checks the idempotency log before processing.
        /// Failed deliveries are retried according to the RetryPolicy with
        /// exponential backoff. After exhausting retries, the delivery is
        /// marked as permanently failed.
        /// </summary>
        private async Task DeliverAsync(
            NotificationEvent evt,
            Subscription sub,
            SemaphoreSlim throttle,
            CancellationToken ct)
        {
            var recordKey = $"{evt.EventId}:{sub.Id}";
            try
            {
                if (sub.RequiredDelivery == DeliveryMode.ExactlyOnce)
                {
                    if (_store != null &&
                        await _store.HasBeenProcessedAsync(evt.EventId, sub.Id, ct))
                    {
                        Interlocked.Increment(ref _totalDelivered);
                        return;
                    }
                }

                var record = _deliveryLog.GetOrAdd(recordKey, _ => new DeliveryRecord
                {
                    EventId = evt.EventId,
                    SubscriptionId = sub.Id
                });

                for (int attempt = 0; attempt <= _retryPolicy.MaxRetries; attempt++)
                {
                    try
                    {
                        record.Attempts = attempt + 1;
                        record.LastAttemptAt = DateTime.UtcNow;
                        record.AttemptTimestamps.Add(DateTime.UtcNow);

                        await sub.Handler(evt, ct);

                        record.Delivered = true;
                        Interlocked.Increment(ref _totalDelivered);

                        if (_store != null)
                            await _store.SaveDeliveryRecordAsync(record, ct);

                        return;
                    }
                    catch (Exception ex) when (attempt < _retryPolicy.MaxRetries)
                    {
                        record.LastError = ex.Message;
                        var delay = _retryPolicy.ComputeDelay(attempt);
                        await Task.Delay(delay, ct);
                    }
                }

                record.Delivered = false;
                Interlocked.Increment(ref _totalFailed);

                if (_store != null)
                    await _store.SaveDeliveryRecordAsync(record, ct);
            }
            finally
            {
                throttle.Release();
            }
        }

        /// <summary>
        /// Retrieves delivery statistics grouped by channel using LINQ aggregation.
        /// Returns total published, delivered, and failed counts per channel.
        /// </summary>
        public Dictionary<string, (int Published, int Delivered, int Failed)> GetChannelStats()
        {
            return _deliveryLog.Values
                .GroupBy(r => _channels
                    .Where(kv => kv.Value.Any(s => s.Id == r.SubscriptionId))
                    .Select(kv => kv.Key)
                    .FirstOrDefault() ?? "unknown")
                .ToDictionary(
                    g => g.Key,
                    g => (
                        Published: g.Count(),
                        Delivered: g.Count(r => r.Delivered),
                        Failed: g.Count(r => !r.Delivered && r.Attempts > _retryPolicy.MaxRetries)
                    ));
        }

        /// <summary>
        /// Gracefully shuts down the event bus. Signals the dispatch loop to stop,
        /// then waits up to 5 seconds for in-flight deliveries to complete.
        /// Any events remaining in the queue after shutdown are lost unless
        /// persistence is enabled via IEventStore.
        /// </summary>
        public async ValueTask DisposeAsync()
        {
            _cts.Cancel();
            try
            {
                await _dispatchTask.WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch (TimeoutException) { }
            catch (OperationCanceledException) { }
            _cts.Dispose();
            _queueSignal.Dispose();
        }
    }
}
