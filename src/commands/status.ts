/**
 * ABOUTME: Status command for ralph-tui.
 * Displays information about any existing resumable session.
 */

import {
  hasPersistedSession,
  loadPersistedSession,
  getSessionSummary,
  isSessionResumable,
} from '../session/index.js';

/**
 * Format duration in human-readable form
 */
function formatDuration(startedAt: string, updatedAt: string): string {
  const start = new Date(startedAt).getTime();
  const end = new Date(updatedAt).getTime();
  const durationMs = end - start;

  const seconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

/**
 * Format date for display
 */
function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString();
}

/**
 * Execute the status command
 */
export async function executeStatusCommand(args: string[]): Promise<void> {
  // Parse --cwd option
  let cwd = process.cwd();
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--cwd' && args[i + 1]) {
      cwd = args[i + 1];
      break;
    }
  }

  // Check for session
  const hasSession = await hasPersistedSession(cwd);

  if (!hasSession) {
    console.log('No session found.');
    console.log('');
    console.log('Start a new session with: ralph-tui run');
    return;
  }

  // Load session
  const session = await loadPersistedSession(cwd);
  if (!session) {
    console.log('No session found.');
    return;
  }

  const summary = getSessionSummary(session);
  const resumable = isSessionResumable(session);

  // Display session info
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('                    Ralph TUI Session Status                    ');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Status with icon
  const statusIcon = getStatusIcon(summary.status);
  console.log(`  Status:          ${statusIcon} ${summary.status.toUpperCase()}`);

  // Session details
  console.log(`  Session ID:      ${summary.sessionId.slice(0, 8)}...`);
  console.log(`  Started:         ${formatDate(summary.startedAt)}`);
  console.log(`  Last Updated:    ${formatDate(summary.updatedAt)}`);
  console.log(`  Duration:        ${formatDuration(summary.startedAt, summary.updatedAt)}`);
  console.log('');

  // Progress
  const progressPercent = summary.totalTasks > 0
    ? Math.round((summary.tasksCompleted / summary.totalTasks) * 100)
    : 0;
  const progressBar = createProgressBar(progressPercent, 30);

  console.log('  Progress:');
  console.log(`    ${progressBar} ${progressPercent}%`);
  console.log(`    Tasks: ${summary.tasksCompleted}/${summary.totalTasks} complete`);
  console.log(`    Iteration: ${summary.currentIteration}${summary.maxIterations > 0 ? `/${summary.maxIterations}` : ''}`);
  console.log('');

  // Configuration
  console.log('  Configuration:');
  console.log(`    Agent:         ${summary.agentPlugin}`);
  console.log(`    Tracker:       ${summary.trackerPlugin}`);
  if (summary.epicId) {
    console.log(`    Epic:          ${summary.epicId}`);
  }
  if (summary.prdPath) {
    console.log(`    PRD:           ${summary.prdPath}`);
  }
  console.log('');

  // Iteration history summary
  if (session.iterations.length > 0) {
    console.log('  Recent Iterations:');
    const recentIterations = session.iterations.slice(-5);
    for (const iter of recentIterations) {
      const iterStatus = getIterationStatusIcon(iter.status);
      const duration = Math.round(iter.durationMs / 1000);
      console.log(
        `    ${iterStatus} Iteration ${iter.iteration}: ${iter.taskTitle.slice(0, 40)}${iter.taskTitle.length > 40 ? '...' : ''} (${duration}s)`
      );
    }
    if (session.iterations.length > 5) {
      console.log(`    ... and ${session.iterations.length - 5} more`);
    }
    console.log('');
  }

  // Skipped tasks
  if (session.skippedTaskIds.length > 0) {
    console.log(`  Skipped Tasks: ${session.skippedTaskIds.length}`);
    console.log('');
  }

  // Actions
  console.log('───────────────────────────────────────────────────────────────');
  if (resumable) {
    console.log('  This session can be resumed.');
    console.log('');
    console.log('  To resume:  ralph-tui resume');
    console.log('  To restart: ralph-tui run --force');
  } else {
    console.log('  This session is complete.');
    console.log('');
    console.log('  To start new: ralph-tui run');
  }
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');
}

/**
 * Get status icon
 */
function getStatusIcon(status: string): string {
  switch (status) {
    case 'running':
      return '▶';
    case 'paused':
      return '⏸';
    case 'completed':
      return '✓';
    case 'failed':
      return '✗';
    case 'interrupted':
      return '⊘';
    default:
      return '○';
  }
}

/**
 * Get iteration status icon
 */
function getIterationStatusIcon(status: string): string {
  switch (status) {
    case 'completed':
      return '✓';
    case 'failed':
      return '✗';
    case 'interrupted':
      return '⊘';
    case 'skipped':
      return '⊖';
    default:
      return '○';
  }
}

/**
 * Create a progress bar string
 */
function createProgressBar(percent: number, width: number): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  return `[${'\u2588'.repeat(filled)}${'\u2591'.repeat(empty)}]`;
}

/**
 * Print status command help
 */
export function printStatusHelp(): void {
  console.log(`
ralph-tui status - Check session status

Usage: ralph-tui status [options]

Options:
  --cwd <path>      Working directory (default: current directory)

Description:
  Shows information about any existing Ralph session including:
  - Current status (running, paused, completed, etc.)
  - Progress (tasks completed, current iteration)
  - Configuration (agent, tracker, epic/prd)
  - Recent iteration history
  - Whether the session can be resumed

Examples:
  ralph-tui status              # Check session in current directory
  ralph-tui status --cwd /path  # Check session in specific directory
`);
}
