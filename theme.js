// Apply the saved theme before paint; storage can be unavailable in private contexts.
(() => {
  let theme;
  try { theme = localStorage.theme; } catch { /* use system preference */ }
  document.documentElement.classList.toggle("dark",
    theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches));
  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme");
    const paint = () => {
      const dark = document.documentElement.classList.contains("dark");
      button.textContent = dark ? "light" : "dark";
      button.setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} theme`);
    };
    button.onclick = () => {
      const dark = document.documentElement.classList.toggle("dark");
      try { localStorage.theme = dark ? "dark" : "light"; } catch { /* optional */ }
      paint();
    };
    paint();
  });
})();
