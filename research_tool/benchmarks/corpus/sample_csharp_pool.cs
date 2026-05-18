// ConnectionPool<T> implements a thread-safe, generic connection pool
// with configurable lifecycle management, health checking, and automatic
// eviction of stale connections.
//
// The pool maintains between MinSize and MaxSize connections, creating
// new ones on demand and destroying idle connections that exceed
// MaxIdleTime. A background health check task validates connections
// at regular intervals using the IPooledConnection.IsHealthy property.
//
// Connection acquisition follows a two-phase strategy: first attempt
// to retrieve from the idle queue (O(1) via ConcurrentQueue), then
// create a new connection if below MaxSize. If at capacity, the caller
// blocks on a SemaphoreSlim until a connection is returned.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace ConnectionPooling
{
    /// <summary>
    /// Lifecycle states for pooled connections. Transitions are strictly
    /// one-directional: Created → Idle → Active → (Idle | Evicted).
    /// A connection that fails a health check transitions directly to Evicted.
    /// </summary>
    public enum ConnectionState
    {
        Created = 0,
        Idle = 1,
        Active = 2,
        Evicted = 3,
        Faulted = 4
    }

    /// <summary>
    /// Pool sizing strategy that determines how the pool manages capacity.
    /// Elastic grows and shrinks based on demand, maintaining between MinSize
    /// and MaxSize connections. Fixed maintains exactly MaxSize connections
    /// at all times, pre-warming on startup. Lazy starts empty and grows
    /// to MaxSize but never shrinks below the high-water mark.
    /// </summary>
    public enum PoolSizingStrategy
    {
        Elastic = 0,
        Fixed = 1,
        Lazy = 2
    }

    /// <summary>
    /// Configuration for the connection pool. All durations use TimeSpan
    /// for clarity. The validation rules enforce:
    /// - MinSize >= 0 and MinSize <= MaxSize
    /// - MaxSize >= 1 and MaxSize <= 1000
    /// - MaxIdleTime >= 10 seconds
    /// - HealthCheckInterval >= 5 seconds
    /// - AcquireTimeout >= 100 milliseconds
    /// </summary>
    public record PoolConfiguration
    {
        public int MinSize { get; init; } = 2;
        public int MaxSize { get; init; } = 20;
        public TimeSpan MaxIdleTime { get; init; } = TimeSpan.FromMinutes(5);
        public TimeSpan HealthCheckInterval { get; init; } = TimeSpan.FromSeconds(30);
        public TimeSpan AcquireTimeout { get; init; } = TimeSpan.FromSeconds(10);
        public PoolSizingStrategy SizingStrategy { get; init; } = PoolSizingStrategy.Elastic;
        public int MaxLifetimeSeconds { get; init; } = 3600;
        public bool ValidateOnAcquire { get; init; } = true;
        public bool ValidateOnReturn { get; init; } = false;

        public void Validate()
        {
            if (MinSize < 0)
                throw new ArgumentException($"MinSize must be >= 0, got {MinSize}");
            if (MaxSize < 1 || MaxSize > 1000)
                throw new ArgumentException($"MaxSize must be 1..1000, got {MaxSize}");
            if (MinSize > MaxSize)
                throw new ArgumentException($"MinSize ({MinSize}) must be <= MaxSize ({MaxSize})");
            if (MaxIdleTime < TimeSpan.FromSeconds(10))
                throw new ArgumentException($"MaxIdleTime must be >= 10s, got {MaxIdleTime}");
            if (HealthCheckInterval < TimeSpan.FromSeconds(5))
                throw new ArgumentException($"HealthCheckInterval must be >= 5s, got {HealthCheckInterval}");
            if (AcquireTimeout < TimeSpan.FromMilliseconds(100))
                throw new ArgumentException($"AcquireTimeout must be >= 100ms, got {AcquireTimeout}");
        }
    }

    /// <summary>
    /// Metadata tracked for each pooled connection instance. CreatedAt and
    /// LastUsedAt use Stopwatch.GetTimestamp() for monotonic timing that
    /// is immune to system clock adjustments. UseCount tracks the total
    /// number of times the connection has been acquired from the pool.
    /// </summary>
    public record ConnectionMetadata
    {
        public long CreatedAtTimestamp { get; init; }
        public long LastUsedAtTimestamp { get; set; }
        public long LastHealthCheckTimestamp { get; set; }
        public int UseCount { get; set; }
        public ConnectionState State { get; set; } = ConnectionState.Created;
        public string? LastError { get; set; }

        public TimeSpan Age => Stopwatch.GetElapsedTime(CreatedAtTimestamp);
        public TimeSpan IdleTime => Stopwatch.GetElapsedTime(LastUsedAtTimestamp);
    }

    /// <summary>
    /// Contract for objects managed by the connection pool. Implementations
    /// must provide a synchronous health check and support both sync and
    /// async disposal. The Reset method is called when a connection is
    /// returned to the pool to restore it to a clean state.
    /// </summary>
    public interface IPooledConnection : IAsyncDisposable, IDisposable
    {
        string ConnectionId { get; }
        bool IsHealthy { get; }
        void Reset();
    }

    /// <summary>
    /// Factory interface for creating new pooled connections. The factory
    /// is invoked whenever the pool needs to grow. Implementations should
    /// handle connection establishment, authentication, and initial
    /// configuration. The CancellationToken allows aborting slow connection
    /// setup during pool shutdown.
    /// </summary>
    public interface IConnectionFactory<T> where T : class, IPooledConnection
    {
        Task<T> CreateAsync(CancellationToken ct = default);
    }

    /// <summary>
    /// Snapshot of pool statistics at a point in time. Computed from
    /// the pool's internal state using lock-free reads of atomic counters.
    /// HitRate measures the ratio of successful idle-queue retrievals
    /// to total acquisition attempts, indicating pool efficiency.
    /// </summary>
    public record PoolStatistics
    {
        public int TotalConnections { get; init; }
        public int IdleConnections { get; init; }
        public int ActiveConnections { get; init; }
        public long TotalAcquired { get; init; }
        public long TotalCreated { get; init; }
        public long TotalEvicted { get; init; }
        public long TotalFailed { get; init; }
        public long HealthChecksPassed { get; init; }
        public long HealthChecksFailed { get; init; }
        public double HitRate { get; init; }
        public TimeSpan OldestConnectionAge { get; init; }
    }

    /// <summary>
    /// RAII wrapper returned by AcquireAsync that ensures the connection is
    /// returned to the pool when disposed. Implements IAsyncDisposable so
    /// callers can use "await using" for automatic return.
    ///
    /// The wrapper tracks whether the connection was explicitly returned
    /// via Dispose. If the wrapper is garbage collected without disposal,
    /// the connection is evicted rather than returned, since its state
    /// is unknown.
    /// </summary>
    public sealed class PooledConnectionLease<T> : IAsyncDisposable where T : class, IPooledConnection
    {
        private readonly ConnectionPool<T> _pool;
        private readonly T _connection;
        private int _disposed;

        internal PooledConnectionLease(ConnectionPool<T> pool, T connection)
        {
            _pool = pool;
            _connection = connection;
        }

        public T Connection => _disposed == 0
            ? _connection
            : throw new ObjectDisposedException(nameof(PooledConnectionLease<T>));

        public async ValueTask DisposeAsync()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                await _pool.ReturnAsync(_connection);
            }
        }
    }

    /// <summary>
    /// Thread-safe generic connection pool with lifecycle management.
    ///
    /// Acquisition strategy:
    /// 1. Try to dequeue from the idle channel (non-blocking)
    /// 2. If no idle connections and below MaxSize, create a new one
    /// 3. If at capacity, wait on the idle channel up to AcquireTimeout
    ///
    /// The pool uses an unbounded Channel&lt;T&gt; as its idle connection queue,
    /// providing async-ready FIFO ordering with backpressure. The capacity
    /// semaphore limits the total number of connections (idle + active)
    /// to MaxSize.
    ///
    /// Health checking runs on a periodic timer. Unhealthy connections
    /// are evicted and replaced if the pool is below MinSize. The health
    /// check is non-blocking — it reads the idle channel, checks each
    /// connection, and writes back only healthy connections.
    /// </summary>
    public sealed class ConnectionPool<T> : IAsyncDisposable where T : class, IPooledConnection
    {
        private readonly IConnectionFactory<T> _factory;
        private readonly PoolConfiguration _config;
        private readonly Channel<T> _idleChannel;
        private readonly ConcurrentDictionary<string, ConnectionMetadata> _metadata = new();
        private readonly SemaphoreSlim _capacitySemaphore;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _healthCheckTask;
        private long _totalAcquired;
        private long _totalCreated;
        private long _totalEvicted;
        private long _totalFailed;
        private long _hits;
        private long _misses;
        private long _healthPassed;
        private long _healthFailed;

        public ConnectionPool(IConnectionFactory<T> factory, PoolConfiguration? config = null)
        {
            _config = config ?? new PoolConfiguration();
            _config.Validate();
            _factory = factory;
            _idleChannel = Channel.CreateUnbounded<T>(new UnboundedChannelOptions
            {
                SingleWriter = false,
                SingleReader = false,
            });
            _capacitySemaphore = new SemaphoreSlim(_config.MaxSize, _config.MaxSize);
            _healthCheckTask = Task.Run(() => HealthCheckLoopAsync(_cts.Token));
        }

        /// <summary>
        /// Pre-warms the pool by creating MinSize connections. Called
        /// automatically for the Fixed sizing strategy during construction.
        /// For Elastic and Lazy strategies, call explicitly if desired.
        /// Returns the number of connections successfully created.
        /// </summary>
        public async Task<int> WarmupAsync(CancellationToken ct = default)
        {
            int created = 0;
            for (int i = 0; i < _config.MinSize; i++)
            {
                try
                {
                    var conn = await CreateConnectionAsync(ct);
                    await _idleChannel.Writer.WriteAsync(conn, ct);
                    UpdateMetadata(conn, ConnectionState.Idle);
                    created++;
                }
                catch (Exception)
                {
                    Interlocked.Increment(ref _totalFailed);
                }
            }
            return created;
        }

        /// <summary>
        /// Acquires a connection from the pool, wrapped in a lease for automatic return.
        /// Follows the three-phase acquisition strategy: check idle queue, create new,
        /// or wait for return. The acquire timeout starts when no idle connection is
        /// available and the pool is at capacity.
        ///
        /// When ValidateOnAcquire is true, connections retrieved from the idle queue
        /// are health-checked before being handed out. Unhealthy connections are
        /// evicted and the acquisition retries from phase 1.
        /// </summary>
        public async Task<PooledConnectionLease<T>> AcquireAsync(CancellationToken ct = default)
        {
            Interlocked.Increment(ref _totalAcquired);

            while (true)
            {
                if (_idleChannel.Reader.TryRead(out var idle))
                {
                    Interlocked.Increment(ref _hits);

                    if (_config.ValidateOnAcquire && !idle.IsHealthy)
                    {
                        await EvictAsync(idle);
                        continue;
                    }

                    var meta = _metadata.GetOrAdd(idle.ConnectionId, _ => NewMetadata());
                    if (_config.MaxLifetimeSeconds > 0 &&
                        meta.Age.TotalSeconds > _config.MaxLifetimeSeconds)
                    {
                        await EvictAsync(idle);
                        continue;
                    }

                    meta.State = ConnectionState.Active;
                    meta.LastUsedAtTimestamp = Stopwatch.GetTimestamp();
                    meta.UseCount++;
                    return new PooledConnectionLease<T>(this, idle);
                }

                Interlocked.Increment(ref _misses);

                if (_capacitySemaphore.CurrentCount > 0)
                {
                    if (await _capacitySemaphore.WaitAsync(0, ct))
                    {
                        try
                        {
                            var conn = await CreateConnectionAsync(ct);
                            var meta = _metadata.GetOrAdd(conn.ConnectionId, _ => NewMetadata());
                            meta.State = ConnectionState.Active;
                            meta.UseCount = 1;
                            return new PooledConnectionLease<T>(this, conn);
                        }
                        catch
                        {
                            _capacitySemaphore.Release();
                            throw;
                        }
                    }
                }

                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
                timeout.CancelAfter(_config.AcquireTimeout);

                try
                {
                    await _idleChannel.Reader.WaitToReadAsync(timeout.Token);
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    throw new TimeoutException(
                        $"Could not acquire connection within {_config.AcquireTimeout}");
                }
            }
        }

        /// <summary>
        /// Returns a connection to the pool. If ValidateOnReturn is enabled,
        /// the connection is health-checked before being placed back in the
        /// idle queue. The connection's Reset method is always called to
        /// restore it to a clean state before reuse.
        /// </summary>
        internal async Task ReturnAsync(T connection)
        {
            if (_config.ValidateOnReturn && !connection.IsHealthy)
            {
                await EvictAsync(connection);
                return;
            }

            connection.Reset();

            if (_metadata.TryGetValue(connection.ConnectionId, out var meta))
            {
                meta.State = ConnectionState.Idle;
                meta.LastUsedAtTimestamp = Stopwatch.GetTimestamp();
            }

            await _idleChannel.Writer.WriteAsync(connection);
        }

        /// <summary>
        /// Periodic health check loop that runs every HealthCheckInterval.
        /// Reads all connections from the idle channel, validates each one,
        /// and writes back only healthy connections. Evicted connections
        /// release their capacity semaphore slot for new connections.
        ///
        /// For Elastic strategy, also evicts connections that have been
        /// idle longer than MaxIdleTime, but only if the pool is above MinSize.
        /// </summary>
        private async Task HealthCheckLoopAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(_config.HealthCheckInterval, ct);
                }
                catch (OperationCanceledException) { break; }

                var toCheck = new List<T>();
                while (_idleChannel.Reader.TryRead(out var conn))
                {
                    toCheck.Add(conn);
                }

                int currentTotal = _metadata.Count(m => m.Value.State != ConnectionState.Evicted);
                foreach (var conn in toCheck)
                {
                    if (!conn.IsHealthy)
                    {
                        Interlocked.Increment(ref _healthFailed);
                        await EvictAsync(conn);
                        currentTotal--;
                        continue;
                    }

                    Interlocked.Increment(ref _healthPassed);

                    if (_metadata.TryGetValue(conn.ConnectionId, out var meta))
                    {
                        meta.LastHealthCheckTimestamp = Stopwatch.GetTimestamp();

                        if (_config.SizingStrategy == PoolSizingStrategy.Elastic &&
                            meta.IdleTime > _config.MaxIdleTime &&
                            currentTotal > _config.MinSize)
                        {
                            await EvictAsync(conn);
                            currentTotal--;
                            continue;
                        }
                    }

                    await _idleChannel.Writer.WriteAsync(conn, ct);
                }

                if (_config.SizingStrategy != PoolSizingStrategy.Lazy &&
                    currentTotal < _config.MinSize)
                {
                    int deficit = _config.MinSize - currentTotal;
                    for (int i = 0; i < deficit; i++)
                    {
                        try
                        {
                            if (await _capacitySemaphore.WaitAsync(0, ct))
                            {
                                var conn = await CreateConnectionAsync(ct);
                                await _idleChannel.Writer.WriteAsync(conn, ct);
                                UpdateMetadata(conn, ConnectionState.Idle);
                            }
                        }
                        catch (Exception)
                        {
                            Interlocked.Increment(ref _totalFailed);
                        }
                    }
                }
            }
        }

        private async Task<T> CreateConnectionAsync(CancellationToken ct)
        {
            var conn = await _factory.CreateAsync(ct);
            Interlocked.Increment(ref _totalCreated);
            _metadata[conn.ConnectionId] = NewMetadata();
            return conn;
        }

        private async Task EvictAsync(T connection)
        {
            if (_metadata.TryGetValue(connection.ConnectionId, out var meta))
            {
                meta.State = ConnectionState.Evicted;
            }
            Interlocked.Increment(ref _totalEvicted);
            _capacitySemaphore.Release();

            try
            {
                await connection.DisposeAsync();
            }
            catch { }
        }

        private static ConnectionMetadata NewMetadata() => new()
        {
            CreatedAtTimestamp = Stopwatch.GetTimestamp(),
            LastUsedAtTimestamp = Stopwatch.GetTimestamp(),
        };

        private void UpdateMetadata(T connection, ConnectionState state)
        {
            var meta = _metadata.GetOrAdd(connection.ConnectionId, _ => NewMetadata());
            meta.State = state;
            meta.LastUsedAtTimestamp = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Returns a snapshot of the pool's current statistics. All counters
        /// are read atomically but the snapshot is not transactional — individual
        /// fields may reflect slightly different points in time under concurrent use.
        /// </summary>
        public PoolStatistics GetStatistics()
        {
            long acquired = Interlocked.Read(ref _totalAcquired);
            long hits = Interlocked.Read(ref _hits);
            long created = Interlocked.Read(ref _totalCreated);
            long evicted = Interlocked.Read(ref _totalEvicted);

            var activeMeta = _metadata.Values
                .Where(m => m.State != ConnectionState.Evicted)
                .ToList();

            var oldest = activeMeta.Any()
                ? activeMeta.Max(m => m.Age)
                : TimeSpan.Zero;

            return new PoolStatistics
            {
                TotalConnections = activeMeta.Count,
                IdleConnections = activeMeta.Count(m => m.State == ConnectionState.Idle),
                ActiveConnections = activeMeta.Count(m => m.State == ConnectionState.Active),
                TotalAcquired = acquired,
                TotalCreated = created,
                TotalEvicted = evicted,
                TotalFailed = Interlocked.Read(ref _totalFailed),
                HealthChecksPassed = Interlocked.Read(ref _healthPassed),
                HealthChecksFailed = Interlocked.Read(ref _healthFailed),
                HitRate = acquired > 0 ? (double)hits / acquired : 0.0,
                OldestConnectionAge = oldest,
            };
        }

        /// <summary>
        /// Gracefully shuts down the pool. Stops the health check loop, then
        /// drains and disposes all idle connections. Active connections held by
        /// callers are not forcibly reclaimed — they will be disposed when their
        /// lease completes. The drain timeout is 3x the health check interval
        /// to allow the final health check cycle to complete.
        /// </summary>
        public async ValueTask DisposeAsync()
        {
            _cts.Cancel();
            try
            {
                var drainTimeout = _config.HealthCheckInterval * 3;
                await _healthCheckTask.WaitAsync(drainTimeout);
            }
            catch (TimeoutException) { }
            catch (OperationCanceledException) { }

            _idleChannel.Writer.Complete();

            await foreach (var conn in _idleChannel.Reader.ReadAllAsync())
            {
                try { await conn.DisposeAsync(); }
                catch { }
            }

            _cts.Dispose();
            _capacitySemaphore.Dispose();
        }
    }

    /// <summary>
    /// Extension methods for PoolStatistics providing derived metrics
    /// and formatted output for monitoring systems.
    /// </summary>
    public static class PoolStatisticsExtensions
    {
        /// <summary>
        /// Calculates the connection churn rate as evictions per creation.
        /// A high churn rate (> 0.3) indicates the pool is frequently destroying
        /// and recreating connections, suggesting MaxIdleTime is too aggressive
        /// or the health check is too strict.
        /// </summary>
        public static double ChurnRate(this PoolStatistics stats) =>
            stats.TotalCreated > 0 ? (double)stats.TotalEvicted / stats.TotalCreated : 0.0;

        /// <summary>
        /// Returns the health check pass rate as a percentage.
        /// A pass rate below 95% indicates infrastructure instability
        /// or connection configuration issues.
        /// </summary>
        public static double HealthCheckPassRate(this PoolStatistics stats)
        {
            long total = stats.HealthChecksPassed + stats.HealthChecksFailed;
            return total > 0 ? (double)stats.HealthChecksPassed / total * 100.0 : 100.0;
        }

        /// <summary>
        /// Formats pool statistics as a single-line summary suitable for
        /// structured logging: "pool[total=X idle=Y active=Z hit=H% churn=C%]"
        /// </summary>
        public static string ToLogLine(this PoolStatistics stats)
        {
            return $"pool[total={stats.TotalConnections} idle={stats.IdleConnections} " +
                   $"active={stats.ActiveConnections} hit={stats.HitRate:P1} " +
                   $"churn={stats.ChurnRate():P1}]";
        }
    }
}
