import React, { useState, useEffect, useRef } from "react";
import { ThemeProvider } from "./components/UI/ThemeProvider.jsx";
import { ToastProvider } from "./components/UI/ToastProvider.jsx";
import { ErrorBoundary } from "./components/UI/ErrorBoundary.jsx";
import { SettingsModal } from "./components/Settings/SettingsModal.jsx";
import { useUIStore } from "./store/useUIStore.js";
import { useLibraryStore } from "./store/useLibraryStore.js";
import { useSettingsStore } from "./store/useSettingsStore.js";
import {
  getSessionById,
  getSharedSessionIdFromLocation,
} from "./lib/library.js";
import {
  buildAppViewHref,
  getAppViewFromLocation,
  normalizeAppView,
} from "./lib/appRoutes.js";
import SearchEngine from "./SearchEngine.jsx";
import Library from "./Library.jsx";
import Docs from "./Docs.jsx";
import {
  BookOpen,
  Menu,
  MessageSquare,
  Library as LibraryIcon,
  Settings,
  Plus,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Moon,
  Sun,
} from "lucide-react";

/**
 * Main App Component
 * Root component with navigation and routing
 */
export default function App() {
  const {
    currentRoute,
    sidebar,
    setActiveTab,
    toggleSidebar,
    setSidebarOpen,
    setSidebarCollapsed,
    openModal,
    isMobile,
    setIsMobile,
  } = useUIStore();
  const { loadSessions, clearSelectedSession } = useLibraryStore();
  const theme = useSettingsStore((state) => state.theme);
  const toggleTheme = useSettingsStore((state) => state.toggleTheme);
  const [selectedSession, setSelectedSession] = useState(null);
  const [chatResetToken, setChatResetToken] = useState(0);
  const sidebarRef = useRef(null);
  const lastChatSessionIdRef = useRef("");

  const currentView = normalizeAppView(currentRoute);

  const replaceRouteUrl = React.useCallback(
    (view = "chat", sessionId = null) => {
      if (typeof window === "undefined") return;
      const nextUrl = buildAppViewHref(view, {
        sessionId: normalizeAppView(view) === "chat" ? sessionId : "",
      });
      if (!nextUrl) return;
      window.history.replaceState({}, "", nextUrl);
    },
    [],
  );

  const pushRouteUrl = React.useCallback((view = "chat", sessionId = null) => {
    if (typeof window === "undefined") return;
    const nextUrl = buildAppViewHref(view, {
      sessionId: normalizeAppView(view) === "chat" ? sessionId : "",
    });
    if (!nextUrl) return;
    const currentHref = `${window.location.pathname}${window.location.search}`;
    if (nextUrl === currentHref) return;
    window.history.pushState({}, "", nextUrl);
  }, []);

  const syncRouteFromLocation = React.useCallback(() => {
    if (typeof window === "undefined") return false;

    const nextView = getAppViewFromLocation(window.location);
    setActiveTab(nextView);
    const sessionId = getSharedSessionIdFromLocation(window.location);
    const canonicalUrl = buildAppViewHref(nextView, {
      sessionId: nextView === "chat" ? sessionId : "",
    });
    const currentHref = `${window.location.pathname}${window.location.search}`;
    if (canonicalUrl && canonicalUrl !== currentHref) {
      window.history.replaceState({}, "", canonicalUrl);
    }
    if (nextView !== "chat") {
      clearSelectedSession();
      setSelectedSession(null);
      return false;
    }
    lastChatSessionIdRef.current = sessionId || "";
    if (!sessionId) {
      clearSelectedSession();
      setSelectedSession(null);
      setChatResetToken((current) => current + 1);
      return false;
    }

    const storedSession = getSessionById(sessionId);
    if (!storedSession) {
      clearSelectedSession();
      setSelectedSession(null);
      lastChatSessionIdRef.current = "";
      setChatResetToken((current) => current + 1);
      replaceRouteUrl("chat", null);
      return false;
    }

    setSelectedSession(storedSession);
    return true;
  }, [clearSelectedSession, replaceRouteUrl, setActiveTab]);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    syncRouteFromLocation();

    const handlePopState = () => {
      syncRouteFromLocation();
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [syncRouteFromLocation]);

  // Handle mobile detection and auto-collapse
  useEffect(() => {
    const checkResponsive = () => {
      const width = window.innerWidth;
      const mobile = width < 768;
      const tablet = width >= 768 && width < 1024;

      window.__nubagent_is_mobile = mobile;
      setIsMobile(mobile);

      // Auto-collapse sidebar on tablet
      if (tablet && !sidebar.isCollapsed) {
        setSidebarCollapsed(true);
      }
      // Auto-expand on desktop
      if (width >= 1024 && sidebar.isCollapsed) {
        setSidebarCollapsed(false);
      }
    };

    checkResponsive();
    window.addEventListener("resize", checkResponsive);
    return () => window.removeEventListener("resize", checkResponsive);
  }, [setIsMobile, setSidebarCollapsed, sidebar.isCollapsed]);

  // Handle body class for sidebar state
  useEffect(() => {
    if (isMobile && sidebar.isOpen) {
      document.body.classList.add("sidebar-open");
    } else {
      document.body.classList.remove("sidebar-open");
    }
  }, [sidebar.isOpen, isMobile]);

  // Close sidebar when clicking outside on mobile
  useEffect(() => {
    if (!isMobile || !sidebar.isOpen) return;

    const handleClickOutside = (event) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
        setSidebarOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isMobile, sidebar.isOpen, setSidebarOpen]);

  const closeSidebarForMobile = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const handleNewChat = () => {
    setActiveTab("chat");
    clearSelectedSession();
    setSelectedSession(null);
    setChatResetToken((current) => current + 1);
    lastChatSessionIdRef.current = "";
    pushRouteUrl("chat", null);
    closeSidebarForMobile();
  };

  const handleViewSession = (session) => {
    setSelectedSession(session);
    setActiveTab("chat");
    lastChatSessionIdRef.current = session?.id || "";
    pushRouteUrl("chat", session?.id || null);
    closeSidebarForMobile();
  };

  const handleNavigate = (view) => {
    const nextView = normalizeAppView(view);
    setActiveTab(nextView);
    if (nextView === "library") {
      loadSessions();
    }
    let nextSessionId = null;
    if (nextView === "chat") {
      const restoredSession = lastChatSessionIdRef.current
        ? getSessionById(lastChatSessionIdRef.current)
        : null;
      setSelectedSession(restoredSession || null);
      nextSessionId = restoredSession?.id || null;
      if (!restoredSession) {
        lastChatSessionIdRef.current = "";
      }
    } else {
      clearSelectedSession();
      setSelectedSession(null);
    }
    pushRouteUrl(nextView, nextView === "chat" ? nextSessionId : null);
    closeSidebarForMobile();
  };

  const navItems = [
    {
      id: "chat",
      label: "Research",
      detail: "Search and synthesis",
      icon: <MessageSquare size={18} />,
    },
    {
      id: "library",
      label: "Library",
      detail: "Saved sessions",
      icon: <LibraryIcon size={18} />,
    },
    {
      id: "docs",
      label: "Docs",
      detail: "System guide",
      icon: <BookOpen size={18} />,
    },
  ];
  const currentSessionLabel =
    selectedSession?.title ||
    selectedSession?.query ||
    selectedSession?.name ||
    null;
  const shellSessionLabel = currentView === "chat" ? currentSessionLabel : null;
  const viewMeta = {
    chat: {
      eyebrow: "Research",
      title: "Research",
      summary: currentSessionLabel
        ? `Session loaded: ${currentSessionLabel}`
        : "Search the web and keep the source trail attached.",
    },
    library: {
      eyebrow: "Library",
      title: "Library",
      summary: "Search, reopen, and export saved sessions quickly.",
    },
    docs: {
      eyebrow: "Docs",
      title: "Docs",
      summary: "Reference the runtime, API surface, and research system.",
    },
  };
  const activeViewMeta = viewMeta[currentView] || viewMeta.chat;
  const themeLabel =
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme";

  return (
    <ThemeProvider>
      <ErrorBoundary>
        <div className={`app app-shell app-shell--${currentView}`}>
          {isMobile && sidebar.isOpen && (
            <button
              className="app-shell__backdrop"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close navigation"
            />
          )}
          {/* Sidebar */}
          <aside
            ref={sidebarRef}
            className={`sidebar ${isMobile && sidebar.isOpen ? "sidebar--open" : ""} ${sidebar.isCollapsed ? "sidebar--collapsed" : ""}`}
          >
            <div className="sidebar__header">
              {!sidebar.isCollapsed && (
                <button className="sidebar__new-chat" onClick={handleNewChat}>
                  <Plus size={16} />
                  <span>New chat</span>
                </button>
              )}
              <div className="sidebar__brand">
                <div className="sidebar__logo">
                  <span className="sidebar__logo-icon">
                    <Sparkles size={16} strokeWidth={2.3} />
                  </span>
                  {!sidebar.isCollapsed && (
                    <div className="sidebar__logo-copy">
                      <span className="sidebar__logo-text">NubAgent</span>
                      <span className="sidebar__logo-meta">
                        {shellSessionLabel || activeViewMeta.title}
                      </span>
                    </div>
                  )}
                </div>
              </div>
              {!isMobile && (
                <button
                  className="sidebar__collapse"
                  onClick={() => setSidebarCollapsed(!sidebar.isCollapsed)}
                  aria-label={
                    sidebar.isCollapsed ? "Expand sidebar" : "Collapse sidebar"
                  }
                >
                  {sidebar.isCollapsed ? (
                    <ChevronRight size={16} />
                  ) : (
                    <ChevronLeft size={16} />
                  )}
                </button>
              )}
            </div>

            <nav className="sidebar__nav">
              {!sidebar.isCollapsed && (
                <div className="sidebar__nav-label">Workspace</div>
              )}
              {navItems.map((item) => (
                <button
                  key={item.id}
                  className={`sidebar__nav-item ${currentView === item.id ? "sidebar__nav-item--active" : ""}`}
                  onClick={() => handleNavigate(item.id)}
                  aria-current={currentView === item.id ? "page" : undefined}
                >
                  <span className="sidebar__nav-icon">{item.icon}</span>
                  {!sidebar.isCollapsed && (
                    <span className="sidebar__nav-copy">
                      <strong>{item.label}</strong>
                      <small>{item.detail}</small>
                    </span>
                  )}
                </button>
              ))}
            </nav>

            <div className="sidebar__footer">
              <button
                className="sidebar__footer-item"
                onClick={() => openModal("settings")}
              >
                <Settings size={18} />
                {!sidebar.isCollapsed && (
                  <span className="sidebar__footer-copy">
                    <strong>Settings</strong>
                    <small>Keys, defaults, interface</small>
                  </span>
                )}
              </button>
              <button
                className="sidebar__footer-item"
                onClick={toggleTheme}
                aria-label={themeLabel}
              >
                {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
                {!sidebar.isCollapsed && (
                  <span className="sidebar__footer-copy">
                    <strong>
                      {theme === "dark" ? "Light mode" : "Dark mode"}
                    </strong>
                    <small>
                      {theme === "dark"
                        ? "Switch to light theme"
                        : "Switch to dark theme"}
                    </small>
                  </span>
                )}
              </button>
            </div>
          </aside>

          {/* Mobile Header */}
          {isMobile && (
            <header className="app__mobile-header">
              <button
                className="app__menu-btn"
                onClick={toggleSidebar}
                aria-label={sidebar.isOpen ? "Close menu" : "Open menu"}
              >
                <Menu size={24} />
              </button>
              <div className="app__mobile-brand">
                <span className="app__mobile-brandmark">
                  <Sparkles size={15} strokeWidth={2.3} />
                </span>
                <div className="app__mobile-titleblock">
                  <span className="app__title">NubAgent</span>
                  <span className="app__mobile-detail">
                    {activeViewMeta.title}
                  </span>
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
                <button
                  className="app__mobile-cta"
                  onClick={handleNewChat}
                  aria-label="Start a new chat"
                >
                  <Plus size={18} />
                </button>
              </div>
            </header>
          )}

          {/* Main Content */}
          <main className="app__main">
            {!isMobile && (
              <header className="app__desktop-header">
                <div className="app__desktop-intro">
                  <button className="app__desktop-model" type="button">
                    <span>
                      {currentView === "chat"
                        ? "NubAgent"
                        : activeViewMeta.title}
                    </span>
                  </button>
                  {!sidebar.isCollapsed && (
                    <p className="app__desktop-copy">
                      {currentView === "chat" && currentSessionLabel
                        ? currentSessionLabel
                        : activeViewMeta.summary}
                    </p>
                  )}
                </div>
                <div className="app__desktop-actions">
                  <button
                    className="app__desktop-icon"
                    onClick={toggleTheme}
                    aria-label={themeLabel}
                  >
                    {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
                  </button>
                  <button
                    className="app__desktop-icon"
                    onClick={handleNewChat}
                    aria-label="Start a new chat"
                  >
                    <Plus size={16} />
                  </button>
                </div>
              </header>
            )}
            <div className={`app__main-frame app__main-frame--${currentView}`}>
              <div className="app__main-frame-inner">
                {currentView === "chat" && (
                  <SearchEngine
                    session={selectedSession}
                    resetSignal={chatResetToken}
                    onSessionRouteChange={(sessionId = "") => {
                      lastChatSessionIdRef.current = String(
                        sessionId || "",
                      ).trim();
                    }}
                  />
                )}
                {currentView === "library" && (
                  <Library
                    onViewSession={handleViewSession}
                    onBack={() => handleNavigate("chat")}
                    onNewSearch={handleNewChat}
                  />
                )}
                {currentView === "docs" && <Docs />}
              </div>
            </div>
          </main>

          {/* Settings Modal */}
          <SettingsModal />

          {/* Toast Container */}
          <ToastProvider />
        </div>
      </ErrorBoundary>
    </ThemeProvider>
  );
}
