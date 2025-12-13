import { useState, useEffect } from 'react';
import en from './translations/en';
import fr from './translations/fr';
import ja from './translations/ja';
import zh from './translations/zh';
import es from './translations/es';
import it from './translations/it';
import de from './translations/de';

// Available translations
const translations = {
    en,
    fr,
    ja,
    zh,
    es,
    it,
    de
};

// Language metadata
export const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧', nativeName: 'English' },
    { code: 'fr', name: 'French', flag: '🇫🇷', nativeName: 'Français' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵', nativeName: '日本語' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳', nativeName: '中文' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸', nativeName: 'Español' },
    { code: 'it', name: 'Italian', flag: '🇮🇹', nativeName: 'Italiano' },
    { code: 'de', name: 'German', flag: '🇩🇪', nativeName: 'Deutsch' }
];

const STORAGE_KEY = 'helios-language';
const DEFAULT_LANGUAGE = 'en';

/**
 * Detect browser language and return supported language code
 */
export function detectBrowserLanguage() {
    const browserLang = navigator.language || navigator.userLanguage;
    const langCode = browserLang.split('-')[0].toLowerCase();

    // Check if detected language is supported
    if (translations[langCode]) {
        return langCode;
    }

    return DEFAULT_LANGUAGE;
}

/**
 * Get stored language preference or detect from browser
 */
export function getInitialLanguage() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && translations[stored]) {
            return stored;
        }
    } catch (e) {
        console.warn('Failed to read language from localStorage:', e);
    }

    return detectBrowserLanguage();
}

/**
 * Save language preference to localStorage
 */
export function saveLanguage(langCode) {
    try {
        localStorage.setItem(STORAGE_KEY, langCode);
    } catch (e) {
        console.warn('Failed to save language to localStorage:', e);
    }
}

/**
 * Get translation for a key path (e.g., 'toolbar.runPipeline')
 * Falls back to English if translation not found
 */
export function translate(langCode, keyPath) {
    const keys = keyPath.split('.');
    let value = translations[langCode];

    // Navigate through nested object
    for (const key of keys) {
        if (value && typeof value === 'object' && key in value) {
            value = value[key];
        } else {
            // Fallback to English
            value = translations[DEFAULT_LANGUAGE];
            for (const k of keys) {
                if (value && typeof value === 'object' && k in value) {
                    value = value[k];
                } else {
                    console.warn(`Translation missing: ${langCode}.${keyPath}`);
                    return keyPath; // Return key path as fallback
                }
            }
            break;
        }
    }

    return typeof value === 'string' ? value : keyPath;
}

/**
 * React hook for using translations in components
 */
export function useTranslation() {
    const [language, setLanguageState] = useState(getInitialLanguage());

    const setLanguage = (langCode) => {
        if (translations[langCode]) {
            setLanguageState(langCode);
            saveLanguage(langCode);
        }
    };

    const t = (keyPath) => translate(language, keyPath);

    return { t, language, setLanguage, languages };
}

export default { useTranslation, translate, getInitialLanguage, languages };
