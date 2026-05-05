export enum ColorMode {
	LIGHT = 'light',
	DARK = 'dark',
	SYSTEM = 'system',
	KDE = 'kde'
}

/**
 * MCP prompt display variant
 */
export enum McpPromptVariant {
	MESSAGE = 'message',
	ATTACHMENT = 'attachment'
}

/**
 * URL prefixes for protocol detection
 */
export enum UrlProtocol {
	DATA = 'data:',
	HTTP = 'http://',
	HTTPS = 'https://',
	WEBSOCKET = 'ws://',
	WEBSOCKET_SECURE = 'wss://'
}
