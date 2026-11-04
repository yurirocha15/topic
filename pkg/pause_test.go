package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"
)

func TestRunTUIDoesNotRedrawOnTicksWhilePaused(t *testing.T) {
	app, config, state := newTestTUI()
	state.ui.Paused = true
	var draws atomic.Int32
	app.SetAfterDrawFunc(func(tcell.Screen) { draws.Add(1) })
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- runTUI(ctx, app, config, state, func() {}) }()

	// The refresh interval is 1 ms, so 100 ms covers many ticks.
	time.Sleep(100 * time.Millisecond)
	cancel()
	waitForRunTUI(t, done)
	if got := draws.Load(); got > 3 {
		t.Fatalf("expected only the initial draws while paused, got %d", got)
	}
}
