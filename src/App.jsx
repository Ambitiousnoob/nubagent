import React, { useState, useEffect } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from './lib/queryClient.js';
import { ThemeProvider } from './components/UI/ThemeProvider.jsx';
import { ToastProvider } from './components/UI/ToastProvider.jsx';
import { ErrorBoundary } from './components/UI/ErrorBoundary.jsx';
import { SettingsModal } from './components/Settings/SettingsModal.jsx';
import { useUIStore } from './store/useUIStore.js';
import { useLibraryStore } from './store/useLibraryStore.js';
import SearchEngine from './SearchEngine.jsx';
import Library from './Library.jsx';
import {
  Menu,
  X,
  MessageSquare,
  Library as LibraryIcon,
  Settings,
  Plus,
  Search,
  Home,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

/**
 * Main App Component
 * Root component with navigation and routing
 */
export default function App() {
  const { sidebar, setActiveTab, toggleSidebar, setSidebarCollapsed, openModal, isMobile } = useUIStore();
  const { loadSessions, selectSession, clearSelectedSession } = useLibraryStore();
  const [currentView, setCurrentView] = useState('chat');
  const [selectedSession, setSelectedSession] = useState(null);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Handle mobile detection
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      // Store in a way that components can access
      window.__nubagent_is_mobile = mobile;
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const handleNewChat = () => {
    setCurrentView('chat');
    setActiveTab('chat');
    clearSelectedSession();
    setSelectedSession(null);
    // Trigger new chat in SearchEngine via custom event
    window.dispatchEvent(new CustomEvent('nubagent:new-chat'));
  };

  const handleViewSession = (session) => {
    setSelectedSession(session);
    setCurrentView('chat');
    setActiveTab('chat');
    // Trigger session load via custom event
    window.dispatchEvent(new CustomEvent('nubagent:load-session', { detail: session }));
  };

  const handleNavigate = (view) => {
    setCurrentView(view);
    setActiveTab(view);
    if (view === 'library') {
      loadSessions();
    }
  };

  const navItems = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare size={20} /> },
    { id: 'library', label: 'Library', icon: <LibraryIcon size={20} /> },
  ];

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <ErrorBoundary>
          <div className="app">
            {/* Sidebar */}
            <aside className={`sidebar ${sidebar.isOpen ? 'sidebar--open' : 'sidebar--closed'} ${sidebar.isCollapsed ? 'sidebar--collapsed' : ''}`}>
              <div className="sidebar__header">
                <div className="sidebar__logo">
                  <span className="sidebar__logo-icon">🤖</span>
                  {!sidebar.isCollapsed && <span className="sidebar__logo-text">nubagent</span>}
                </div>
                {!isMobile && (
                  <button
                    className="sidebar__collapse"
                    onClick={() => setSidebarCollapsed(!sidebar.isCollapsed)}
                    aria-label={sidebar.isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                  >
                    {sidebar.isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                  </button>
                )}
                {isMobile && (
                  <button
                    className="sidebar__close"
                    onClick={toggleSidebar}
                    aria-label="Close sidebar"
                  >
                    <X size={20} />
                  </button>
                )}
              </div>

              <nav className="sidebar__nav">
                {navItems.map((item) => (
                  <button
                    key={item.id}
                    className={`sidebar__nav-item ${currentView === item.id ? 'sidebar__nav-item--active' : ''}`}
                    onClick={() => handleNavigate(item.id)}
                  >
                    {item.icon}
                    {!sidebar.isCollapsed && <span>{item.label}</span>}
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
                <button
                  className="sidebar__footer-item"
                  onClick={() => openModal('settings')}
                >
                  <Settings size={18} />
                  {!sidebar.isCollapsed && <span>Settings</span>}
                </button>
              </div>
            </aside>

            {/* Mobile Header */}
            {isMobile && (
              <header className="app__mobile-header">
                <button
                  className="app__menu-btn"
                  onClick={toggleSidebar}
                  aria-label="Open menu"
                >
                  <Menu size={24} />
                </button>
                <span className="app__title">nubagent</span>
                <div className="app__mobile-spacer" />
              </header>
            )}

            {/* Main Content */}
            <main className="app__main">
              {currentView === 'chat' && (
                <SearchEngine
                  session={selectedSession}
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
            </main>

            {/* Settings Modal */}
            <SettingsModal />

            {/* Toast Container */}
            <ToastProvider />
          </div>
        </ErrorBoundary>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
