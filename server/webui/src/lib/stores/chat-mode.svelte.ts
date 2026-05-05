/**
 * chat-mode store — selects how much conversation history is sent to the model.
 *
 *  - 'Dialogue' (default): full history is sent (subject to --max-history on the
 *    server). Multi-turn conversation; the model sees its prior replies.
 *  - 'QA': only the last user turn (plus any system message) is sent. Each
 *    question is independent — the model has no memory of prior turns within
 *    the conversation.
 *
 * The conversation in IndexedDB and on screen is unaffected; this only changes
 * what ChatService sends in the request body.
 */

import { browser } from '$app/environment';

export type ChatMode = 'QA' | 'Dialogue';

const STORAGE_KEY = 'chat-mode';
const DEFAULT_MODE: ChatMode = 'Dialogue';

function loadFromStorage(): ChatMode {
	if (!browser) return DEFAULT_MODE;
	const saved = localStorage.getItem(STORAGE_KEY);
	return saved === 'QA' || saved === 'Dialogue' ? saved : DEFAULT_MODE;
}

class ChatModeStore {
	mode = $state<ChatMode>(loadFromStorage());

	set(m: ChatMode): void {
		this.mode = m;
		if (browser) {
			localStorage.setItem(STORAGE_KEY, m);
		}
	}
}

export const chatModeStore = new ChatModeStore();

/** Convenience: read current mode without subscribing. */
export const chatMode = (): ChatMode => chatModeStore.mode;
