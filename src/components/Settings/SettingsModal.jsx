import React from "react";
import { Modal } from "../UI/Modal.jsx";
import { ApiKeyForm } from "./ApiKeyForm.jsx";
import { ThemeToggle } from "./ThemeToggle.jsx";
import { ModelSelector } from "./ModelSelector.jsx";
import { Button } from "../UI/Button.jsx";
import { useSettingsStore } from "../../store/useSettingsStore.js";
import { useUIStore } from "../../store/useUIStore.js";
import {
  Settings,
  Key,
  Moon,
  Cpu,
  Bell,
  Trash2,
  RotateCcw,
} from "lucide-react";

/**
 * SettingsModal Component
 * Main settings modal container
 */
export function SettingsModal() {
  const { modals, closeModal } = useUIStore();
  const { clearAllSettings } = useSettingsStore();
  const [activeTab, setActiveTab] = React.useState("general");
  const [showClearConfirm, setShowClearConfirm] = React.useState(false);

  const tabs = [
    { id: "general", label: "General", icon: <Settings size={16} /> },
    { id: "api", label: "API Keys", icon: <Key size={16} /> },
    { id: "model", label: "Research Models", icon: <Cpu size={16} /> },
    { id: "notifications", label: "Notifications", icon: <Bell size={16} /> },
  ];

  const handleClearAll = () => {
    clearAllSettings();
    setShowClearConfirm(false);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case "general":
        return (
          <div className="settings-tab">
            <div className="settings-section">
              <h3 className="settings-section__title">Appearance</h3>
              <div className="settings-item">
                <div className="settings-item__label">
                  <Moon size={18} />
                  <span>Theme</span>
                </div>
                <ThemeToggle />
              </div>
            </div>

            <div className="settings-section">
              <h3 className="settings-section__title">Data</h3>
              <div className="settings-item">
                <div className="settings-item__label">
                  <Trash2 size={18} />
                  <span>Clear all settings</span>
                </div>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => setShowClearConfirm(true)}
                >
                  Clear
                </Button>
              </div>
            </div>
          </div>
        );

      case "api":
        return (
          <div className="settings-tab">
            <ApiKeyForm />
          </div>
        );

      case "model":
        return (
          <div className="settings-tab">
            <ModelSelector />
          </div>
        );

      case "notifications":
        return (
          <div className="settings-tab">
            <div className="settings-section">
              <h3 className="settings-section__title">Notifications</h3>
              <p className="settings-section__description">
                Configure notification preferences
              </p>
              <div className="settings-item">
                <div className="settings-item__label">
                  <span>Toast notifications</span>
                </div>
                <input
                  type="checkbox"
                  defaultChecked
                  className="settings-toggle"
                />
              </div>
              <div className="settings-item">
                <div className="settings-item__label">
                  <span>Sound effects</span>
                </div>
                <input type="checkbox" className="settings-toggle" />
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <>
      <Modal
        isOpen={modals.settings}
        onClose={() => closeModal("settings")}
        title="Settings"
        size="lg"
        className="settings-modal"
      >
        <div className="settings-modal__layout">
          <div className="settings-modal__tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`settings-modal__tab ${activeTab === tab.id ? "settings-modal__tab--active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
          <div className="settings-modal__content">{renderTabContent()}</div>
        </div>
      </Modal>

      <Modal
        isOpen={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        title="Clear All Settings"
        size="sm"
      >
        <div className="settings-clear-confirm">
          <p>
            Are you sure you want to clear all settings? This action cannot be
            undone.
          </p>
          <div className="settings-clear-confirm__actions">
            <Button
              variant="outline"
              onClick={() => setShowClearConfirm(false)}
            >
              Cancel
            </Button>
            <Button variant="danger" onClick={handleClearAll}>
              <RotateCcw size={16} />
              Clear All
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}

export default SettingsModal;
