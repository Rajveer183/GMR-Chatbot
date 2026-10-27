import { lookup, STATUS_KEYS } from "./uiStrings";
import type { UIKey } from "./uiStrings";

/** Response languages for the header dropdown. The codes match the backend's
 * gmr_agents/language.py LANGUAGES; "auto" answers in the language of each question. */
export type LanguageCode = "en" | "hi" | "hinglish" | "bn" | "mr" | "ta" | "te" | "gu" | "pa" | "kn" | "ml";
export type LanguagePreference = "auto" | LanguageCode;

export const LANGUAGE_OPTIONS: { value: LanguagePreference; label: string }[] = [
  { value: "auto", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi (हिंदी)" },
  { value: "hinglish", label: "Hinglish" },
  { value: "bn", label: "Bengali (বাংলা)" },
  { value: "mr", label: "Marathi (मराठी)" },
  { value: "ta", label: "Tamil (தமிழ்)" },
  { value: "te", label: "Telugu (తెలుగు)" },
  { value: "gu", label: "Gujarati (ગુજરાતી)" },
  { value: "pa", label: "Punjabi (ਪੰਜਾਬੀ)" },
  { value: "kn", label: "Kannada (ಕನ್ನಡ)" },
  { value: "ml", label: "Malayalam (മലയാളം)" },
];

const PREFERENCES = new Set<string>(LANGUAGE_OPTIONS.map((o) => o.value));

// Kept across messages and reloads; localStorage can throw (private mode,
// blocked site data) -- then the choice just lasts until the page closes.
const LANGUAGE_KEY = "gmr-chat.responseLanguage";

export function readStoredLanguage(): LanguagePreference {
  try {
    const stored = window.localStorage.getItem(LANGUAGE_KEY);
    return stored && PREFERENCES.has(stored) ? (stored as LanguagePreference) : "auto";
  } catch {
    return "auto";
  }
}

export function storeLanguage(language: LanguagePreference) {
  try {
    window.localStorage.setItem(LANGUAGE_KEY, language);
  } catch {
    // Non-fatal -- see readStoredLanguage.
  }
}

/** BCP 47 tag for the `lang` attribute (correct font shaping and screen-reader voice). */
export function langTag(language: string | null | undefined): string {
  if (language === "hinglish") return "hi-Latn";
  return language && PREFERENCES.has(language) && language !== "auto" ? language : "en";
}

/** Voice-input locale for the browser's speech recognition. */
export function speechLocale(language: LanguagePreference): string {
  if (language === "auto" || language === "en") return "en-US";
  if (language === "hinglish") return "en-IN";
  return `${language}-IN`;
}

/** Screen text in a language ({name} placeholders filled from `vars`); English when the
 * language is unknown. The strings themselves live in uiStrings.ts. */
export function uiText(language: string | null | undefined, key: UIKey, vars?: Record<string, string | number>): string {
  let text = lookup(language || "en", key);
  for (const [name, value] of Object.entries(vars ?? {})) text = text.split(`{${name}}`).join(String(value));
  return text;
}

/** A backend progress message ("Running the query") in the screen language. */
export function statusText(language: string, message: string): string {
  const key = STATUS_KEYS[message];
  return key ? uiText(language, key) : message;
}

/** The language the screen itself is shown in: the dropdown's when one is picked, else the
 * language of the latest answer (Auto-detect follows the conversation). */
export function screenLanguage(preference: LanguagePreference, latestAnswerLanguage?: string | null): string {
  if (preference !== "auto") return preference;
  return latestAnswerLanguage && PREFERENCES.has(latestAnswerLanguage) ? latestAnswerLanguage : "en";
}
