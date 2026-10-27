import React from "react";
import { ChevronDown, Languages } from "lucide-react";
import { LANGUAGE_OPTIONS } from "../i18n/languages";
import type { LanguagePreference } from "../i18n/languages";

interface LanguageSelectProps {
  value: LanguagePreference;
  onChange: (language: LanguagePreference) => void;
  /** "Answer language" in the screen language. */
  label?: string;
}

// A native <select>: keyboard navigation, screen-reader support and the phone's own
// picker come for free; only the closed control is styled to match the header.
export const LanguageSelect: React.FC<LanguageSelectProps> = ({ value, onChange, label = "Answer language" }) => (
  <label className="chat-language-select" title={label}>
    <Languages size={14} aria-hidden="true" />
    <span className="chat-visually-hidden">{label}</span>
    <select value={value} onChange={(e) => onChange(e.target.value as LanguagePreference)}>
      {LANGUAGE_OPTIONS.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
    <ChevronDown size={13} aria-hidden="true" className="chat-language-chevron" />
  </label>
);
