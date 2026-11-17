// Light / dark / system theme, chosen from the profile menu (ProfileMenu.tsx).
// The resolved theme is written to <html data-theme="light|dark">; theme.css swaps
// its tokens on [data-theme="dark"]. index.html applies the saved choice before
// first paint so a dark user never sees a light flash.

export type ThemePref = "system" | "light" | "dark";

const KEY = "gmr.theme";
const media = () =>
  typeof window !== "undefined" && window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

export function readThemePref(): ThemePref {
  try {
    const v = window.localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

function apply(pref: ThemePref) {
  const dark = pref === "dark" || (pref === "system" && !!media()?.matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

export function setThemePref(pref: ThemePref) {
  try {
    if (pref === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, pref);
  } catch {
    /* storage blocked: the choice still applies for this page */
  }
  apply(pref);
}

/** Apply the saved choice and follow the OS while it is "system". */
export function installTheme() {
  apply(readThemePref());
  media()?.addEventListener?.("change", () => {
    if (readThemePref() === "system") apply("system");
  });
}
