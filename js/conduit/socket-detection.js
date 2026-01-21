/**
 * Conduit Socket Detection Module
 * DOM-based socket detection using data-slot-key attributes
 */

/**
 * Parse a slot key into its components.
 * Format: "nodeId-in-slotIndex" or "nodeId-out-slotIndex"
 * Examples: "13-in-0", "13-out-0", "83-in-1"
 */
export function parseSlotKey(slotKey) {
    if (!slotKey) return null;

    // Format: nodeId-in-index or nodeId-out-index
    const parts = slotKey.split('-');
    if (parts.length !== 3) {
        return null;
    }

    const [nodeIdStr, inOut, indexStr] = parts;
    const nodeId = parseInt(nodeIdStr, 10);
    const slotIndex = parseInt(indexStr, 10);

    if (isNaN(nodeId) || isNaN(slotIndex)) return null;
    if (inOut !== 'in' && inOut !== 'out') return null;

    return {
        nodeId,
        slotIndex,
        side: inOut === 'in' ? "input" : "output",
    };
}

/**
 * Get slot info from click coordinates by finding slot element.
 * Searches up for data-slot-key, or looks inside .lg-slot container.
 */
export function getSlotAtPoint(x, y) {
    const el = document.elementFromPoint(x, y);
    if (!el) return null;

    // First: check if we clicked directly on or inside an element with data-slot-key
    const slotEl = el.closest('[data-slot-key]');
    if (slotEl) {
        return parseSlotKey(slotEl.dataset.slotKey);
    }

    // Second: check if we're inside a .lg-slot container (the data-slot-key may be on a child)
    const slotContainer = el.closest('.lg-slot');
    if (slotContainer) {
        const keyEl = slotContainer.querySelector('[data-slot-key]');
        if (keyEl) {
            return parseSlotKey(keyEl.dataset.slotKey);
        }
    }

    return null;
}
