import { setMode } from 'mode-watcher';
import { ColorMode } from '$lib/enums/ui';

/**
 * Apply a theme to the document.
 *
 * `mode-watcher` only knows light/dark/system. The KDE theme is layered on
 * top: we set mode-watcher to `dark` (so the dark-mode token base is active)
 * and add a `.kde` class on <html>, which the CSS uses to override the
 * palette / radii / font with KDE Breeze values.
 */
export function applyTheme(theme: string): void {
	if (typeof document === 'undefined') return;

	const root = document.documentElement;

	if (theme === ColorMode.KDE) {
		setMode(ColorMode.DARK);
		root.classList.add('kde');
	} else {
		root.classList.remove('kde');
		setMode(theme as ColorMode);
	}
}
