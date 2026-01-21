/**
 * Conduit Logging Module
 * Unified logging with verbosity levels
 *
 * Levels:
 *   0 = OFF      - No logging
 *   1 = MINIMAL  - Key lifecycle events only
 *   2 = NORMAL   - All events with basic data (default)
 *   3 = VERBOSE  - Everything including high-frequency events
 */

// Log level constants
export const LOG_OFF = 0;
export const LOG_MINIMAL = 1;
export const LOG_NORMAL = 2;
export const LOG_VERBOSE = 3;

// Current log level - loaded from config, defaults to NORMAL
let currentLogLevel = LOG_NORMAL;

// Debounce tracking for high-frequency events
const debounceTimers = {};
const DEBOUNCE_MS = 100;

/**
 * Load log level from backend config.
 * Called once during initialization.
 */
export async function initLogging() {
    try {
        const response = await fetch("/conduit/config");
        const config = await response.json();
        if (typeof config.log_level === "number") {
            currentLogLevel = config.log_level;
        }
        console.log(`[Conduit] Log level: ${currentLogLevel} (${getLevelName(currentLogLevel)})`);
    } catch (e) {
        console.log("[Conduit] Using default log level: NORMAL");
    }
}

/**
 * Get human-readable level name.
 */
function getLevelName(level) {
    switch (level) {
        case LOG_OFF: return "OFF";
        case LOG_MINIMAL: return "MINIMAL";
        case LOG_NORMAL: return "NORMAL";
        case LOG_VERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
    }
}

/**
 * Get the current log level.
 */
export function getLogLevel() {
    return currentLogLevel;
}

/**
 * Set the log level programmatically.
 */
export function setLogLevel(level) {
    currentLogLevel = level;
}

/**
 * Send a log entry to the backend.
 *
 * @param {string} type - Event type
 * @param {object} data - Event data
 * @param {number} level - Required log level (default: NORMAL)
 * @param {object} options - Optional settings
 *   - debounce: boolean - If true, debounce rapid repeated calls
 */
export async function logToBackend(type, data = {}, level = LOG_NORMAL, options = {}) {
    // Skip if current level is too low
    if (currentLogLevel < level) {
        return;
    }

    // Handle debouncing for high-frequency events
    if (options.debounce) {
        if (debounceTimers[type]) {
            clearTimeout(debounceTimers[type]);
        }
        debounceTimers[type] = setTimeout(() => {
            delete debounceTimers[type];
            sendLog(type, data, level);
        }, DEBOUNCE_MS);
        return;
    }

    await sendLog(type, data, level);
}

/**
 * Internal: Actually send the log to backend.
 */
async function sendLog(type, data, level) {
    try {
        await fetch("/conduit/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                type,
                data,
                level,
                timestamp: Date.now()
            }),
        });
    } catch (e) {
        // Silent fail - logging shouldn't break the app
    }
}

// =============================================================================
// Convenience functions for each level
// =============================================================================

/**
 * Log at MINIMAL level - key lifecycle events only.
 * Use for: setup complete, workflow saved, execution success/fail
 */
export function logMinimal(type, data = {}) {
    return logToBackend(type, data, LOG_MINIMAL);
}

/**
 * Log at NORMAL level - standard events.
 * Use for: tag toggled, config changed, hook installed
 */
export function logNormal(type, data = {}) {
    return logToBackend(type, data, LOG_NORMAL);
}

/**
 * Log at VERBOSE level - high-frequency or detailed events.
 * Use for: sockets_collected, tooltip updates, style updates
 */
export function logVerbose(type, data = {}, options = {}) {
    return logToBackend(type, data, LOG_VERBOSE, options);
}
