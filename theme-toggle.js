/* Bascule thème clair/sombre — partagée par toutes les pages HTML du projet.
   Doit être chargée en tout début de <head> (avant les <style>) pour poser
   l'attribut data-theme sur <html> avant le premier rendu (évite le flash). */
(function () {
  "use strict";
  var KEY = "pref-theme";

  function getStoredTheme() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function storeTheme(theme) {
    try { localStorage.setItem(KEY, theme); } catch (e) { /* stockage indisponible */ }
  }
  function systemPrefersLight() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  }
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
  }
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }
  function updateButtons(theme) {
    var btns = document.querySelectorAll("[data-theme-toggle-btn]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute("aria-pressed", theme === "light" ? "true" : "false");
      btns[i].setAttribute("title", theme === "light" ? "Passer en mode sombre" : "Passer en mode clair");
    }
  }

  // Applique le thème le plus tôt possible, avant le rendu de <body>.
  var initial = getStoredTheme() || (systemPrefersLight() ? "light" : "dark");
  apply(initial);

  window.toggleTheme = function () {
    var next = currentTheme() === "light" ? "dark" : "light";
    apply(next);
    storeTheme(next);
    updateButtons(next);
  };

  document.addEventListener("DOMContentLoaded", function () {
    var btns = document.querySelectorAll("[data-theme-toggle-btn]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener("click", window.toggleTheme);
    }
    updateButtons(currentTheme());
  });
})();
