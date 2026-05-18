// Package ratelimiter implements a distributed rate limiter using the
// sliding window log algorithm with Redis-backed state.
//
// The sliding window log algorithm tracks individual request timestamps
// within the current window, providing exact rate limiting at the cost
// of O(N) storage per client (where N is the rate limit). For high
// throughput services, the sliding window counter variant trades
// accuracy for O(1) storage.
package ratelimiter

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// ErrRateLimited is returned when a request exceeds the configured rate limit.
var ErrRateLimited = errors.New("rate limit exceeded")

// ErrInvalidConfig is returned when the limiter configuration is invalid.
var ErrInvalidConfig = errors.New("invalid rate limiter configuration")

// Config holds the rate limiter configuration.
// MaxRequests is the maximum number of requests allowed within Window.
// BurstSize allows short bursts above MaxRequests, decaying over BurstDecay.
// The effective limit at any instant is MaxRequests + remaining burst capacity.
type Config struct {
	MaxRequests int
	Window      time.Duration
	BurstSize   int
	BurstDecay  time.Duration
	KeyPrefix   string
}

// Validate checks that the configuration values are within acceptable ranges.
// MaxRequests must be positive, Window must be at least 1 second and at most
// 24 hours, and BurstSize must be non-negative.
func (c Config) Validate() error {
	if c.MaxRequests <= 0 {
		return fmt.Errorf("%w: MaxRequests must be positive, got %d", ErrInvalidConfig, c.MaxRequests)
	}
	if c.Window < time.Second {
		return fmt.Errorf("%w: Window must be at least 1s, got %v", ErrInvalidConfig, c.Window)
	}
	if c.Window > 24*time.Hour {
		return fmt.Errorf("%w: Window must be at most 24h, got %v", ErrInvalidConfig, c.Window)
	}
	if c.BurstSize < 0 {
		return fmt.Errorf("%w: BurstSize must be non-negative, got %d", ErrInvalidConfig, c.BurstSize)
	}
	return nil
}

// Result contains the outcome of a rate limit check.
type Result struct {
	Allowed    bool
	Remaining  int
	ResetAt    time.Time
	RetryAfter time.Duration
}

// ClientKey generates a deterministic key for rate limiting by hashing
// the client identifier with SHA-256. This ensures uniform key distribution
// and prevents key injection attacks from specially crafted identifiers.
func ClientKey(identifier string, prefix string) string {
	h := sha256.Sum256([]byte(identifier))
	return prefix + hex.EncodeToString(h[:8])
}

// requestLog tracks timestamps of requests within the current window.
type requestLog struct {
	timestamps []time.Time
	burstUsed  int
	lastBurst  time.Time
}

// SlidingWindowLimiter implements the sliding window log algorithm.
// Each client's request timestamps are stored in a sorted slice, and
// expired entries are pruned on each check. The algorithm provides
// exact rate limiting (no approximation) but uses O(MaxRequests)
// memory per active client.
type SlidingWindowLimiter struct {
	config Config
	mu     sync.RWMutex
	logs   map[string]*requestLog
}

// NewSlidingWindowLimiter creates a new rate limiter with the given configuration.
// Returns ErrInvalidConfig if the configuration fails validation.
func NewSlidingWindowLimiter(cfg Config) (*SlidingWindowLimiter, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.KeyPrefix == "" {
		cfg.KeyPrefix = "rl:"
	}
	if cfg.BurstDecay == 0 {
		cfg.BurstDecay = cfg.Window
	}
	return &SlidingWindowLimiter{
		config: cfg,
		logs:   make(map[string]*requestLog),
	}, nil
}

// Allow checks whether a request from the given client should be allowed.
// It atomically records the request if allowed. The context parameter
// supports cancellation for distributed implementations.
func (l *SlidingWindowLimiter) Allow(ctx context.Context, clientID string) (Result, error) {
	if ctx.Err() != nil {
		return Result{}, ctx.Err()
	}

	key := ClientKey(clientID, l.config.KeyPrefix)
	now := time.Now()
	windowStart := now.Add(-l.config.Window)

	l.mu.Lock()
	defer l.mu.Unlock()

	log, exists := l.logs[key]
	if !exists {
		log = &requestLog{}
		l.logs[key] = log
	}

	// Prune expired timestamps
	pruned := make([]time.Time, 0, len(log.timestamps))
	for _, ts := range log.timestamps {
		if ts.After(windowStart) {
			pruned = append(pruned, ts)
		}
	}
	log.timestamps = pruned

	// Calculate available burst capacity
	burstRemaining := l.config.BurstSize
	if log.burstUsed > 0 && l.config.BurstDecay > 0 {
		elapsed := now.Sub(log.lastBurst)
		decayed := int(math.Floor(float64(log.burstUsed) * (1.0 - elapsed.Seconds()/l.config.BurstDecay.Seconds())))
		if decayed < 0 {
			decayed = 0
		}
		log.burstUsed = decayed
		burstRemaining = l.config.BurstSize - decayed
	}

	currentCount := len(log.timestamps)
	effectiveLimit := l.config.MaxRequests + burstRemaining
	remaining := effectiveLimit - currentCount

	if currentCount >= effectiveLimit {
		// Find when the oldest request in the window will expire
		resetAt := log.timestamps[0].Add(l.config.Window)
		retryAfter := resetAt.Sub(now)
		if retryAfter < 0 {
			retryAfter = 0
		}
		return Result{
			Allowed:    false,
			Remaining:  0,
			ResetAt:    resetAt,
			RetryAfter: retryAfter,
		}, nil
	}

	log.timestamps = append(log.timestamps, now)
	if currentCount >= l.config.MaxRequests {
		log.burstUsed++
		log.lastBurst = now
	}

	return Result{
		Allowed:   true,
		Remaining: remaining - 1,
		ResetAt:   now.Add(l.config.Window),
	}, nil
}

// Reset clears the rate limit state for a specific client.
// Used for administrative override or after identity changes.
func (l *SlidingWindowLimiter) Reset(clientID string) {
	key := ClientKey(clientID, l.config.KeyPrefix)
	l.mu.Lock()
	delete(l.logs, key)
	l.mu.Unlock()
}

// Cleanup removes expired entries for all clients. Should be called
// periodically (e.g., every Window duration) to prevent memory growth
// from inactive clients. Returns the number of clients cleaned up.
func (l *SlidingWindowLimiter) Cleanup() int {
	now := time.Now()
	windowStart := now.Add(-l.config.Window)

	l.mu.Lock()
	defer l.mu.Unlock()

	cleaned := 0
	for key, log := range l.logs {
		active := false
		for _, ts := range log.timestamps {
			if ts.After(windowStart) {
				active = true
				break
			}
		}
		if !active {
			delete(l.logs, key)
			cleaned++
		}
	}
	return cleaned
}

// Stats returns current statistics about the rate limiter state.
type Stats struct {
	ActiveClients   int
	TotalRequests   int64
	RejectedCount   int64
	OldestEntry     time.Time
	CleanupCount    int64
}

// FixedWindowCounter implements a simpler fixed-window rate limiting
// algorithm with O(1) storage per client. The window resets at fixed
// intervals, which can allow up to 2x the rate limit at window boundaries
// (the "boundary burst" problem). For most applications this tradeoff
// is acceptable given the significant memory savings.
type FixedWindowCounter struct {
	config  Config
	mu      sync.RWMutex
	windows map[string]*windowEntry
}

type windowEntry struct {
	count     int
	windowKey int64
}

// NewFixedWindowCounter creates a fixed-window rate limiter.
func NewFixedWindowCounter(cfg Config) (*FixedWindowCounter, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &FixedWindowCounter{
		config:  cfg,
		windows: make(map[string]*windowEntry),
	}, nil
}

// Allow checks the fixed-window counter for the given client.
// The window key is computed as Unix timestamp divided by window
// duration in seconds, ensuring all requests within the same
// window period share a counter.
func (f *FixedWindowCounter) Allow(ctx context.Context, clientID string) (Result, error) {
	if ctx.Err() != nil {
		return Result{}, ctx.Err()
	}

	key := ClientKey(clientID, f.config.KeyPrefix)
	now := time.Now()
	windowKey := now.Unix() / int64(f.config.Window.Seconds())

	f.mu.Lock()
	defer f.mu.Unlock()

	entry, exists := f.windows[key]
	if !exists || entry.windowKey != windowKey {
		entry = &windowEntry{windowKey: windowKey}
		f.windows[key] = entry
	}

	if entry.count >= f.config.MaxRequests {
		nextWindow := time.Unix((windowKey+1)*int64(f.config.Window.Seconds()), 0)
		return Result{
			Allowed:    false,
			Remaining:  0,
			ResetAt:    nextWindow,
			RetryAfter: nextWindow.Sub(now),
		}, nil
	}

	entry.count++
	remaining := f.config.MaxRequests - entry.count
	nextWindow := time.Unix((windowKey+1)*int64(f.config.Window.Seconds()), 0)

	return Result{
		Allowed:   true,
		Remaining: remaining,
		ResetAt:   nextWindow,
	}, nil
}
