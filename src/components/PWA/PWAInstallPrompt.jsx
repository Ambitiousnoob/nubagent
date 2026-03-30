/**
 * PWA Install Prompt Component
 * Shows install prompt for PWA-capable browsers
 */

import React, { useState, useEffect } from 'react';
import { X, Download, Smartphone } from 'lucide-react';
import { Button } from './UI/Button.jsx';

export function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      return;
    }

    // Check if user dismissed before
    const dismissed = localStorage.getItem('pwa-install-dismissed');
    if (dismissed) {
      const dismissedAt = new Date(dismissed);
      const daysSinceDismissal = (Date.now() - dismissedAt.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceDismissal < 7) {
        return;
      }
    }

    // Listen for install prompt
    const handleBeforeInstallPrompt = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      // Show prompt after a delay
      setTimeout(() => setShowPrompt(true), 5000);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;

    if (outcome === 'accepted') {
      console.log('User accepted the install prompt');
    }

    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  const handleDismiss = () => {
    setShowPrompt(false);
    localStorage.setItem('pwa-install-dismissed', new Date().toISOString());
  };

  if (!showPrompt) return null;

  return (
    <div className="pwa-install-prompt">
      <div className="pwa-install-prompt__content">
        <div className="pwa-install-prompt__icon">
          <Smartphone size={32} />
        </div>
        <h3 className="pwa-install-prompt__title">Install NubAgent</h3>
        <p className="pwa-install-prompt__text">
          Install NubAgent for a better experience with offline support and quick access.
        </p>
        <div className="pwa-install-prompt__actions">
          <Button variant="primary" onClick={handleInstall}>
            <Download size={16} />
            Install
          </Button>
          <Button variant="ghost" onClick={handleDismiss}>
            Not Now
          </Button>
        </div>
        <button
          className="pwa-install-prompt__close"
          onClick={handleDismiss}
          aria-label="Close"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}

export default PWAInstallPrompt;
