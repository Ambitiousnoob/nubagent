import React, { useState, useEffect, useCallback } from "react";
import {
    getSavedSessions,
    deleteSession,
    searchSessions,
    formatSessionDate,
} from "./lib/library.js";

const getDomain = (url) => {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

const getFavicon = (url) => {
    try { return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`; } catch { return null; }
};

function LibraryCard({ session, onView, onDelete }) {
    const [isDeleting, setIsDeleting] = useState(false);

    const handleDelete = (e) => {
        e.stopPropagation();
        setIsDeleting(true);
        setTimeout(() => {
            onDelete(session.id);
        }, 200);
    };

    const sourceCount = session.sources?.length || 0;
    const firstSource = session.sources?.[0];

    return (
        <div
            className={`lib-card ${isDeleting ? "lib-card--deleting" : ""}`}
            onClick={() => onView(session)}
        >
            <div className="lib-card__header">
                <div className="lib-card__date">{formatSessionDate(session.createdAt)}</div>
                {sourceCount > 0 && (
                    <div className="lib-card__sources">{sourceCount} source{sourceCount === 1 ? "" : "s"}</div>
                )}
            </div>
            <h3 className="lib-card__query">{session.query}</h3>
            <p className="lib-card__preview">
                {(session.heading || session.body || "").replace(/[#*`\[\]]/g, "").slice(0, 120)}
                {(session.heading || session.body || "").length > 120 ? "..." : ""}
            </p>
            {firstSource && (
                <div className="lib-card__source">
                    {getFavicon(firstSource.url) && (
                        <img
                            src={getFavicon(firstSource.url)}
                            className="lib-card__favicon"
                            alt=""
                            onError={(e) => { e.currentTarget.style.display = "none"; }}
                        />
                    )}
                    <span className="lib-card__domain">{getDomain(firstSource.url)}</span>
                </div>
            )}
            <div className="lib-card__actions">
                <button className="lib-card__view" onClick={() => onView(session)}>Open</button>
                <button className="lib-card__delete" onClick={handleDelete} title="Delete">🗑</button>
            </div>
        </div>
    );
}

function LibraryEmpty({ onNewSearch }) {
    return (
        <div className="lib-empty">
            <div className="lib-empty__icon">📚</div>
            <h2 className="lib-empty__title">Your library is empty</h2>
            <p className="lib-empty__text">
                Saved research sessions will appear here. Start a new search to create your first session.
            </p>
            <button className="lib-empty__btn" onClick={onNewSearch}>
                New Search
            </button>
        </div>
    );
}

function LibrarySearch({ value, onChange, resultCount }) {
    return (
        <div className="lib-search">
            <input
                type="text"
                className="lib-search__input"
                placeholder="Search your library..."
                value={value}
                onChange={(e) => onChange(e.target.value)}
            />
            <span className="lib-search__count">{resultCount} session{resultCount === 1 ? "" : "s"}</span>
        </div>
    );
}

function LibraryConfirmDelete({ sessionName, onConfirm, onCancel }) {
    return (
        <div className="lib-confirm">
            <div className="lib-confirm__content">
                <div className="lib-confirm__icon">⚠️</div>
                <h3 className="lib-confirm__title">Delete this session?</h3>
                <p className="lib-confirm__text">
                    "{sessionName.slice(0, 50)}{sessionName.length > 50 ? "..." : ""}"
                </p>
                <p className="lib-confirm__sub">This action cannot be undone.</p>
            </div>
            <div className="lib-confirm__actions">
                <button className="lib-confirm__cancel" onClick={onCancel}>Cancel</button>
                <button className="lib-confirm__delete" onClick={onConfirm}>Delete</button>
            </div>
        </div>
    );
}

export default function Library({ onBack, onViewSession, onNewSearch }) {
    const [sessions, setSessions] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [deletingId, setDeletingId] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const loadSessions = useCallback(() => {
        setIsLoading(true);
        setTimeout(() => {
            const all = getSavedSessions();
            setSessions(all);
            setIsLoading(false);
        }, 150);
    }, []);

    useEffect(() => {
        loadSessions();
    }, [loadSessions]);

    const filteredSessions = searchQuery.trim()
        ? searchSessions(searchQuery)
        : sessions;

    const handleDelete = useCallback((id) => {
        deleteSession(id);
        setSessions(prev => prev.filter(s => s.id !== id));
        setDeletingId(null);
    }, []);

    const handleView = useCallback((session) => {
        onViewSession?.(session);
    }, [onViewSession]);

    const sessionToDelete = deletingId ? sessions.find(s => s.id === deletingId) : null;

    return (
        <div className="library">
            <style>{`
.library{flex:1;display:flex;flex-direction:column;overflow:hidden;background:var(--bg);padding:24px;max-width:900px;margin:0 auto;width:100%}
.lib-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.lib-title{font-size:20px;font-weight:700;color:var(--tx)}
.lib-back{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-size:13px;cursor:pointer;transition:all .2s}
.lib-back:hover{background:var(--sur);color:var(--tx)}
.lib-search{display:flex;align-items:center;gap:10px;margin-bottom:20px;padding:12px 16px;background:var(--bg3);border:1px solid var(--bdr);border-radius:10px}
.lib-search__input{flex:1;border:none;background:transparent;font-family:var(--font);font-size:14px;color:var(--tx);outline:none}
.lib-search__input::placeholder{color:var(--txm)}
.lib-search__count{font-size:12px;color:var(--txd);white-space:nowrap}
.lib-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;overflow-y:auto;padding:4px}
.lib-card{position:relative;display:flex;flex-direction:column;padding:16px;background:var(--bgc);border:1px solid var(--bdr);border-radius:12px;cursor:pointer;transition:all .2s;min-height:180px}
.lib-card:hover{border-color:rgba(0,201,167,.3);transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3)}
.lib-card--deleting{opacity:0;transform:scale(.95);transition:all .2s}
.lib-card__header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.lib-card__date{font-size:11px;font-weight:600;color:var(--txd);text-transform:uppercase;letter-spacing:.05em}
.lib-card__sources{font-size:11px;font-weight:600;color:var(--ac);background:var(--acd);padding:2px 8px;border-radius:6px}
.lib-card__query{font-size:14px;font-weight:600;color:var(--tx);line-height:1.4;margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.lib-card__preview{font-size:12px;color:var(--txd);line-height:1.6;flex:1;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;margin-bottom:12px}
.lib-card__source{display:flex;align-items:center;gap:6px;padding:6px 8px;background:var(--sur);border-radius:6px;margin-bottom:10px}
.lib-card__favicon{width:14px;height:14px;border-radius:3px}
.lib-card__domain{font-size:11px;color:var(--txd);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px}
.lib-card__actions{display:flex;gap:8px;border-top:1px solid var(--bdr);padding-top:10px;margin-top:auto}
.lib-card__view{flex:1;padding:6px 12px;border-radius:6px;border:none;background:var(--ac);color:#000;font-size:12px;font-weight:600;cursor:pointer;transition:opacity .2s}
.lib-card__view:hover{opacity:.85}
.lib-card__delete{width:32px;height:32px;border-radius:6px;border:none;background:transparent;color:var(--txd);font-size:14px;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center}
.lib-card__delete:hover{background:rgba(255,68,68,.1);color:var(--red)}
.lib-empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 20px}
.lib-empty__icon{font-size:48px;margin-bottom:16px;opacity:.5}
.lib-empty__title{font-size:18px;font-weight:600;color:var(--tx);margin-bottom:8px}
.lib-empty__text{font-size:14px;color:var(--txd);max-width:320px;line-height:1.6;margin-bottom:20px}
.lib-empty__btn{padding:10px 24px;border-radius:8px;border:none;background:var(--ac);color:#000;font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s}
.lib-empty__btn:hover{opacity:.85}
.lib-confirm{position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;z-index:1000;padding:20px}
.lib-confirm__content{background:var(--bg3);border:1px solid var(--bdr);border-radius:14px;padding:24px;max-width:380px;width:100%;text-align:center}
.lib-confirm__icon{font-size:32px;margin-bottom:12px}
.lib-confirm__title{font-size:16px;font-weight:600;color:var(--tx);margin-bottom:8px}
.lib-confirm__text{font-size:13px;color:var(--txd);margin-bottom:16px;word-break:break-word}
.lib-confirm__sub{font-size:12px;color:var(--red);margin-bottom:20px}
.lib-confirm__actions{display:flex;gap:10px}
.lib-confirm__cancel{flex:1;padding:10px 16px;border-radius:8px;border:1px solid var(--bdr);background:transparent;color:var(--tx);font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
.lib-confirm__cancel:hover{background:var(--sur)}
.lib-confirm__delete{flex:1;padding:10px 16px;border-radius:8px;border:none;background:var(--red);color:#fff;font-size:13px;font-weight:600;cursor:pointer;transition:opacity .2s}
.lib-confirm__delete:hover{opacity:.85}
.lib-loading{flex:1;display:flex;align-items:center;justify-content:center}
.lib-loading__spinner{width:32px;height:32px;border:3px solid var(--bdr);border-top-color:var(--ac);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
            `}</style>

            <div className="lib-header">
                <h1 className="lib-title">📚 Library</h1>
                <button className="lib-back" onClick={onBack}>← Back to Search</button>
            </div>

            <LibrarySearch
                value={searchQuery}
                onChange={setSearchQuery}
                resultCount={filteredSessions.length}
            />

            {isLoading ? (
                <div className="lib-loading">
                    <div className="lib-loading__spinner" />
                </div>
            ) : filteredSessions.length === 0 ? (
                searchQuery.trim() ? (
                    <div className="lib-empty">
                        <div className="lib-empty__icon">🔍</div>
                        <h2 className="lib-empty__title">No matches found</h2>
                        <p className="lib-empty__text">
                            Try a different search term or clear the search to see all sessions.
                        </p>
                    </div>
                ) : (
                    <LibraryEmpty onNewSearch={onNewSearch} />
                )
            ) : (
                <div className="lib-grid">
                    {filteredSessions.map((session) => (
                        <LibraryCard
                            key={session.id}
                            session={session}
                            onView={handleView}
                            onDelete={setDeletingId}
                        />
                    ))}
                </div>
            )}

            {sessionToDelete && (
                <LibraryConfirmDelete
                    sessionName={sessionToDelete.query}
                    onConfirm={() => handleDelete(sessionToDelete.id)}
                    onCancel={() => setDeletingId(null)}
                />
            )}
        </div>
    );
}
