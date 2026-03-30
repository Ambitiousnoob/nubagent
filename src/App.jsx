import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider } from './components/UI/ThemeProvider.jsx';
import { ToastProvider } from './components/UI/ToastProvider.jsx';
import { ErrorBoundary } from './components/UI/ErrorBoundary.jsx';
import { SettingsModal } from './components/Settings/SettingsModal.jsx';
import { useUIStore } from './store/useUIStore.js';
import { useLibraryStore } from './store/useLibraryStore.js';
import { buildSessionShareUrl, getSessionById } from './lib/library.js';
import SearchEngine from './SearchEngine.jsx';
import Library from './Library.jsx';
import Docs from './Docs.jsx';
import './styles/app-shell.css';
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
  Compass,
} from 'lucide-react';

/**
 * Main App Component
 * Root component with navigation and routing
 */
export default function App() {
  const { sidebar, setActiveTab, toggleSidebar, setSidebarOpen, setSidebarCollapsed, openModal, isMobile, setIsMobile } = useUIStore();
  const { loadSessions, clearSelectedSession } = useLibraryStore();
  const [currentView, setCurrentView] = useState('chat');
  const [selectedSession, setSelectedSession] = useState(null);
  const [chatResetToken, setChatResetToken] = useState(0);
  const sidebarRef = useRef(null);

  const syncSessionFromLocation = React.useCallback(() => {
    if (typeof window === 'undefined') return false;

    const sessionId = new URLSearchParams(window.location.search).get('session');
    if (!sessionId) {
      setSelectedSession(null);
      return false;
    }

    const storedSession = getSessionById(sessionId);
    if (!storedSession) {
      setSelectedSession(null);
      return false;
    }

    setSelectedSession(storedSession);
    setCurrentView('chat');
    setActiveTab('chat');
    return true;
  }, [setActiveTab]);

  const replaceSessionUrl = (sessionId = null) => {
    if (typeof window === 'undefined') return;

    if (sessionId) {
      window.history.replaceState({}, '', buildSessionShareUrl(sessionId));
      return;
    }

    const url = new URL(window.location.href);
    url.searchParams.delete('session');
    window.history.replaceState({}, '', url.toString());
  };

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    syncSessionFromLocation();

    const handlePopState = () => {
      syncSessionFromLocation();
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [syncSessionFromLocation]);

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
    window.addEventListener('resize', checkResponsive);
    return () => window.removeEventListener('resize', checkResponsive);
  }, [setIsMobile, setSidebarCollapsed, sidebar.isCollapsed]);

  // Handle body class for sidebar state
  useEffect(() => {
    if (isMobile && sidebar.isOpen) {
      document.body.classList.add('sidebar-open');
    } else {
      document.body.classList.remove('sidebar-open');
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

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMobile, sidebar.isOpen, setSidebarOpen]);

  const closeSidebarForMobile = () => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const handleNewChat = () => {
    setCurrentView('chat');
    setActiveTab('chat');
    clearSelectedSession();
    setSelectedSession(null);
    setChatResetToken((current) => current + 1);
    replaceSessionUrl(null);
    closeSidebarForMobile();
  };

  const handleViewSession = (session) => {
    setSelectedSession(session);
    setCurrentView('chat');
    setActiveTab('chat');
    replaceSessionUrl(session?.id || null);
    closeSidebarForMobile();
  };

  const handleNavigate = (view) => {
    setCurrentView(view);
    setActiveTab(view);
    if (view === 'library') {
      loadSessions();
    }
    closeSidebarForMobile();
  };

  const navItems = [
    { id: 'chat', label: 'Research', detail: 'Search and synthesis', icon: <MessageSquare size={18} /> },
    { id: 'library', label: 'Library', detail: 'Saved sessions', icon: <LibraryIcon size={18} /> },
    { id: 'docs', label: 'Docs', detail: 'System guide', icon: <BookOpen size={18} /> },
  ];
  const currentSessionLabel =
    selectedSession?.title ||
    selectedSession?.query ||
    selectedSession?.name ||
    null;
  const viewMeta = {
    chat: {
      eyebrow: 'Research',
      title: 'Research',
      summary: currentSessionLabel
        ? `Session loaded: ${currentSessionLabel}`
        : 'Search the web and keep the source trail attached.',
    },
    library: {
      eyebrow: 'Library',
      title: 'Library',
      summary: 'Search, reopen, and export saved sessions quickly.',
    },
    docs: {
      eyebrow: 'Docs',
      title: 'Docs',
      summary: 'Reference the runtime, API surface, and research system.',
    },
  };
  const activeViewMeta = viewMeta[currentView] || viewMeta.chat;

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
            className={`sidebar ${isMobile && sidebar.isOpen ? 'sidebar--open' : ''} ${sidebar.isCollapsed ? 'sidebar--collapsed' : ''}`}
          >
            <div className="sidebar__header">
              <div className="sidebar__brand">
                <div className="sidebar__logo">
                  <span className="sidebar__logo-icon"><Sparkles size={16} strokeWidth={2.3} /></span>
                  <div className="sidebar__logo-copy">
                    <span className="sidebar__logo-text">nubagent</span>
                    <span className="sidebar__logo-meta">Research workspace</span>
                  </div>
                </div>
                {!sidebar.isCollapsed && (
                  <div className="sidebar__context">
                    <span className="sidebar__context-label">{activeViewMeta.eyebrow}</span>
                    <span className="sidebar__context-separator" aria-hidden="true">•</span>
                    <span className="sidebar__context-value">{currentView === 'chat' ? 'Live' : 'Ready'}</span>
                  </div>
                )}
              </div>
              <button
                className="sidebar__collapse"
                onClick={() => setSidebarCollapsed(!sidebar.isCollapsed)}
                aria-label={sidebar.isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {sidebar.isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
              </button>
            </div>

            <nav className="sidebar__nav">
              {!sidebar.isCollapsed && <div className="sidebar__nav-label">Workspace</div>}
              {navItems.map((item) => (
                <button
                  key={item.id}
                  className={`sidebar__nav-item ${currentView === item.id ? 'sidebar__nav-item--active' : ''}`}
                  onClick={() => handleNavigate(item.id)}
                  aria-current={currentView === item.id ? 'page' : undefined}
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

            <div className="sidebar__actions">
              <button
                className="sidebar__new-chat"
                onClick={handleNewChat}
              >
                <Plus size={18} />
                {!sidebar.isCollapsed && <span>New Chat</span>}
              </button>
            </div>

            <div className="sidebar__footer">
              {!sidebar.isCollapsed && <div className="sidebar__nav-label sidebar__nav-label--footer">Control</div>}
              <button
                className="sidebar__footer-item"
                onClick={() => openModal('settings')}
              >
                <Settings size={18} />
                {!sidebar.isCollapsed && (
                  <span className="sidebar__footer-copy">
                    <strong>Settings</strong>
                    <small>Keys, defaults, interface</small>
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
                aria-label={sidebar.isOpen ? 'Close menu' : 'Open menu'}
              >
                <Menu size={24} />
              </button>
              <div className="app__mobile-brand">
                <span className="app__mobile-brandmark"><Sparkles size={15} strokeWidth={2.3} /></span>
                <div className="app__mobile-titleblock">
                  <span className="app__title">nubagent</span>
                  <span className="app__mobile-detail">{activeViewMeta.title}</span>
                </div>
              </div>
              <button className="app__mobile-cta" onClick={handleNewChat} aria-label="Start a new chat">
                <Compass size={18} />
              </button>
            </header>
          )}

          {/* Main Content */}
          <main className="app__main">
            {!isMobile && (
              <header className="app__desktop-header">
                <div className="app__desktop-intro">
                  <div className="app__desktop-eyebrow">{activeViewMeta.eyebrow}</div>
                  <h1 className="app__desktop-title">{activeViewMeta.title}</h1>
                  <p className="app__desktop-copy">
                    {currentView === 'chat' && currentSessionLabel
                      ? `Working session: ${currentSessionLabel}`
                      : activeViewMeta.summary}
                  </p>
                </div>
                <div className="app__desktop-sidecar">
                  <div className="app__desktop-actions">
                    <button className="app__desktop-btn app__desktop-btn--ghost" onClick={() => openModal('settings')}>
                      <Settings size={16} />
                      <span>Settings</span>
                    </button>
                    <button className="app__desktop-btn app__desktop-btn--primary" onClick={handleNewChat}>
                      <Plus size={16} />
                      <span>New Chat</span>
                    </button>
                  </div>
                </div>
              </header>
            )}
            <div className={`app__main-frame app__main-frame--${currentView}`}>
              <div className="app__main-frame-inner">
                {currentView === 'chat' && (
                  <SearchEngine
                    session={selectedSession}
                    resetSignal={chatResetToken}
                    onSessionLoaded={() => setSelectedSession(null)}
                  />
                )}
                {currentView === 'library' && (
                  <Library
                    onViewSession={handleViewSession}
                    onBack={() => handleNavigate('chat')}
                    onNewSearch={handleNewChat}
                  />
                )}
                {currentView === 'docs' && <Docs />}
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
