import React, { useEffect, useState } from "react";
import { ThemeProvider } from "./components/UI/ThemeProvider.jsx";
import { ToastProvider } from "./components/UI/ToastProvider.jsx";
import { ErrorBoundary } from "./components/UI/ErrorBoundary.jsx";
import { useSettingsStore } from "./store/useSettingsStore.js";
import Docs from "./Docs.jsx";
import {
  BookOpen,
  FileText,
  Layers3,
  Menu,
  Moon,
  Network,
  Sparkles,
  Sun,
  Users,
} from "lucide-react";

const SECTION_LINKS = [
  {
    href: "#top",
    label: "Overview",
    detail: "Landing, purpose, and structure",
    icon: <BookOpen size={18} />,
  },
  {
    href: "#api-reference",
    label: "API Reference",
    detail: "Public paths and owning files",
    icon: <FileText size={18} />,
  },
  {
    href: "#api-creation",
    label: "API Playbook",
    detail: "How new endpoints should ship",
    icon: <Layers3 size={18} />,
  },
  {
    href: "#research-framework",
    label: "Runtime Map",
    detail: "Research DAG and orchestration",
    icon: <Network size={18} />,
  },
  {
    href: "#subagent-catalog",
    label: "Subagents",
    detail: "Operational ownership groups",
    icon: <Users size={18} />,
  },
];

const getViewportState = () => {
  if (typeof window === "undefined") return false;
  return window.innerWidth < 768;
};

const getCurrentHash = () => {
  if (typeof window === "undefined") return "#top";
  return window.location.hash || "#top";
};

export default function App() {
  const theme = useSettingsStore((state) => state.theme);
  const toggleTheme = useSettingsStore((state) => state.toggleTheme);
  const [isMobile, setIsMobile] = useState(getViewportState);
  const [navOpen, setNavOpen] = useState(false);
  const [activeHash, setActiveHash] = useState(getCurrentHash);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const syncViewport = () => {
      const nextIsMobile = window.innerWidth < 768;
      setIsMobile(nextIsMobile);
      if (!nextIsMobile) setNavOpen(false);
    };

    syncViewport();
    window.addEventListener("resize", syncViewport);
    return () => window.removeEventListener("resize", syncViewport);
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;

    if (isMobile && navOpen) {
      document.body.classList.add("sidebar-open");
    } else {
      document.body.classList.remove("sidebar-open");
    }

    return () => document.body.classList.remove("sidebar-open");
  }, [isMobile, navOpen]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const syncHash = () => setActiveHash(getCurrentHash());
    const canonicalizePath = () => {
      const nextUrl = `/${window.location.hash || ""}`;
      const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
      if (currentUrl !== nextUrl) {
        window.history.replaceState({}, "", nextUrl);
      }
      syncHash();
    };

    canonicalizePath();
    window.addEventListener("hashchange", syncHash);
    window.addEventListener("popstate", canonicalizePath);
    return () => {
      window.removeEventListener("hashchange", syncHash);
      window.removeEventListener("popstate", canonicalizePath);
    };
  }, []);

  const themeLabel =
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme";

  const handleSectionClick = (href) => {
    setActiveHash(href);
    if (isMobile) setNavOpen(false);
  };

  return (
    <ThemeProvider>
      <ErrorBoundary>
        <div className="app app-shell app-shell--docs-only" id="top">
          {isMobile && navOpen && (
            <button
              className="app-shell__backdrop"
              onClick={() => setNavOpen(false)}
              aria-label="Close navigation"
            />
          )}

          <aside className={`sidebar ${isMobile && navOpen ? "sidebar--open" : ""}`}>
            <div className="sidebar__header">
              <div className="sidebar__brand">
                <div className="sidebar__logo">
                  <span className="sidebar__logo-icon">
                    <Sparkles size={16} strokeWidth={2.3} />
                  </span>
                  <div className="sidebar__logo-copy">
                    <span className="sidebar__logo-text">NubAgent Docs</span>
                    <span className="sidebar__logo-meta">
                      Documentation-only frontend
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <nav className="sidebar__nav" aria-label="Documentation sections">
              <div className="sidebar__nav-label">Sections</div>
              {SECTION_LINKS.map((item) => {
                const isActive =
                  item.href === "#top"
                    ? activeHash === "#top"
                    : activeHash === item.href;

                return (
                  <a
                    key={item.href}
                    className={`sidebar__nav-item ${isActive ? "sidebar__nav-item--active" : ""}`}
                    href={item.href}
                    onClick={() => handleSectionClick(item.href)}
                    aria-current={isActive ? "location" : undefined}
                  >
                    <span className="sidebar__nav-icon">{item.icon}</span>
                    <span className="sidebar__nav-copy">
                      <strong>{item.label}</strong>
                      <small>{item.detail}</small>
                    </span>
                  </a>
                );
              })}
            </nav>

            <div className="sidebar__footer">
              <button
                className="sidebar__footer-item"
                onClick={toggleTheme}
                aria-label={themeLabel}
              >
                {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
                <span className="sidebar__footer-copy">
                  <strong>{theme === "dark" ? "Light mode" : "Dark mode"}</strong>
                  <small>{themeLabel}</small>
                </span>
              </button>
            </div>
          </aside>

          {isMobile && (
            <header className="app__mobile-header">
              <button
                className="app__menu-btn"
                onClick={() => setNavOpen((current) => !current)}
                aria-label={navOpen ? "Close menu" : "Open menu"}
              >
                <Menu size={24} />
              </button>
              <div className="app__mobile-brand">
                <span className="app__mobile-brandmark">
                  <Sparkles size={15} strokeWidth={2.3} />
                </span>
                <div className="app__mobile-titleblock">
                  <span className="app__title">NubAgent Docs</span>
                  <span className="app__mobile-detail">API and runtime site</span>
                </div>
              </div>
              <div className="app__mobile-actions">
                <button
                  className="app__mobile-cta"
                  onClick={toggleTheme}
                  aria-label={themeLabel}
                >
                  {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
                </button>
              </div>
            </header>
          )}

          <main className="app__main">
            {!isMobile && (
              <header className="app__desktop-header">
                <div className="app__desktop-intro">
                  <button className="app__desktop-model" type="button">
                    <span>Documentation Site</span>
                  </button>
                  <p className="app__desktop-copy">
                    The chat workspace has been removed from the frontend. This
                    build now surfaces the API, runtime, and ownership docs
                    only.
                  </p>
                </div>
                <div className="app__desktop-actions">
                  <button
                    className="app__desktop-icon"
                    onClick={toggleTheme}
                    aria-label={themeLabel}
                  >
                    {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
                  </button>
                </div>
              </header>
            )}

            <div className="app__main-frame app__main-frame--docs">
              <div className="app__main-frame-inner">
                <Docs />
              </div>
            </div>
          </main>

          <ToastProvider />
        </div>
      </ErrorBoundary>
    </ThemeProvider>
  );
}
