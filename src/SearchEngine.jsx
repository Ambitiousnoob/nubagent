import React, { useState, useRef, useEffect, useCallback } from "react";

const CHAT_API = "/api/chat";
const MODEL_NAME = "nub-agent";

const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const getDomain = (url) => {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

const getFavicon = (url) => {
    try { return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`; } catch { return null; }
};

const extractSources = (text = "") => {
    const sources = [];
    const seen = new Set();

    try {
        const parsed = JSON.parse(text);
        const items = Array.isArray(parsed) ? parsed : (parsed?.results || parsed?.organic || []);
        for (const item of items) {
            if (!item?.url || seen.has(item.url)) continue;
            seen.add(item.url);
            sources.push({
                title: item.title || getDomain(item.url),
                url: item.url,
                description: item.description || item.snippet || "",
                date: item.date || null,
            });
        }
    } catch {
        for (const match of String(text || "").matchAll(/https?:\/\/[^\s"',>\]]+/g)) {
            const url = match[0].replace(/[.,;)]+$/, "");
            if (seen.has(url)) continue;
            seen.add(url);
            sources.push({ title: getDomain(url), url, description: "" });
        }
    }

    return sources;
};

const extractToolSources = (tools = []) => {
    const merged = [];
    const seen = new Set();

    for (const tool of Array.isArray(tools) ? tools : []) {
        const candidates = [tool?.args, tool?.note, tool?.result, tool?.content];
        for (const candidate of candidates) {
            const text = typeof candidate === "string" ? candidate : JSON.stringify(candidate || {});
            for (const source of extractSources(text)) {
                if (seen.has(source.url)) continue;
                seen.add(source.url);
                merged.push(source);
            }
        }
    }

    return merged;
};

const extractAnswerParts = (text = "") => {
    const clean = String(text || "").replace(/##\s*Sources?[\s\S]*$/i, "").trim();
    const lines = clean.split("\n");
    const heading = (lines[0] || "").replace(/^#+\s*/, "").trim();
    const body = lines.length > 1 ? lines.slice(1).join("\n").trim() : clean;
    return {
        heading: heading || "Answer",
        body,
    };
};

function Markdown({ text, sources = [] }) {
    const renderInline = (str) => {
        const parts = str.split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g);
        return parts.map((part, index) => {
            if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={index}>{part.slice(2, -2)}</strong>;
            if (/^`[^`]+`$/.test(part)) return <code key={index} className="ic">{part.slice(1, -1)}</code>;
            if (/^\[\d+\]$/.test(part)) {
                const sourceIndex = Number(part.slice(1, -1)) - 1;
                return (
                    <a key={index} href={sources[sourceIndex]?.url || "#"} target="_blank" rel="noopener noreferrer" className="cite">
                        {part.slice(1, -1)}
                    </a>
                );
            }
            return part;
        });
    };

    const lines = String(text || "").split("\n");
    const elements = [];
    let list = [];

    const flush = (key) => {
        if (!list.length) return;
        elements.push(
            <ul key={`ul${key}`} className="md-ul">
                {list.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
            </ul>,
        );
        list = [];
    };

    lines.forEach((line, index) => {
        const heading = line.match(/^(#{1,3})\s(.*)/);
        if (heading) {
            flush(index);
            const level = heading[1].length;
            const content = renderInline(heading[2]);
            if (level === 1) elements.push(<h1 key={index} className="md-h1">{content}</h1>);
            else if (level === 2) elements.push(<h2 key={index} className="md-h2">{content}</h2>);
            else elements.push(<h3 key={index} className="md-h3">{content}</h3>);
            return;
        }

        if (/^[-*] /.test(line)) {
            list.push(line.slice(2));
            return;
        }

        if (!line.trim()) {
            flush(index);
            elements.push(<br key={index} />);
            return;
        }

        flush(index);
        elements.push(<p key={index} className="md-p">{renderInline(line)}</p>);
    });

    flush("end");
    return <>{elements}</>;
}

function SearchingCard({ queries, done }) {
    const [shown, setShown] = useState(0);
    const [dot, setDot] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => setShown((count) => Math.min(count + 1, queries.length)), 550);
        return () => clearInterval(timer);
    }, [queries.length]);

    useEffect(() => {
        if (done) return undefined;
        const timer = setInterval(() => setDot((value) => (value + 1) % 3), 380);
        return () => clearInterval(timer);
    }, [done]);

    return (
        <div className={`scard ${done ? "scard--done" : ""}`}>
            <div className="scard__head">
                <span className="scard__icon">🌐</span>
                <span className="scard__title">{done ? "Searched the web" : "Searching the web"}</span>
            </div>
            {queries.length > 0 && (
                <div className="scard__chips">
                    {queries.slice(0, shown).map((query, index) => (
                        <span key={index} className="qchip" style={{ animationDelay: `${index * 0.08}s` }}>
                            <span className="qchip__dot">🔍</span>
                            {query}
                        </span>
                    ))}
                </div>
            )}
            {!done && (
                <div className="scard__dots">
                    {[0, 1, 2].map((index) => <span key={index} className={`sdot ${dot === index ? "sdot--on" : ""}`}>●</span>)}
                    <span className="scard__searching">Searching...</span>
                </div>
            )}
        </div>
    );
}

const STEPS = [
    ["Identifying", "key search areas and topics"],
    ["Preparing", "autonomous research workflow"],
    ["Analyzing", "query for optimal results"],
    ["Selecting", "search strategies and operators"],
];

function PlanningCard({ done }) {
    const [visible, setVisible] = useState(0);

    useEffect(() => {
        let count = 0;
        const timer = setInterval(() => {
            count += 1;
            setVisible(count);
            if (count >= STEPS.length) clearInterval(timer);
        }, 650);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className={`scard ${done ? "scard--done" : ""}`}>
            <div className="scard__head">
                <span className="scard__icon">💡</span>
                <div>
                    <div className="scard__title">Research Planning</div>
                    <div className="scard__sub">Analyzing requirements</div>
                </div>
            </div>
            <div className="scard__steps">
                {STEPS.map(([strongText, rest], index) => (
                    <div key={index} className={`pstep ${index < visible ? "pstep--done" : "pstep--wait"}`}>
                        <span className="pstep__icon">{index < visible ? "✓" : "○"}</span>
                        <span><strong>{strongText}</strong> <span className="pstep__rest">{rest}...</span></span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function SourcePills({ sources }) {
    if (!sources.length) return null;
    return (
        <div className="pills">
            {sources.slice(0, 6).map((source, index) => (
                <a key={index} href={source.url} target="_blank" rel="noopener noreferrer" className="pill">
                    {getFavicon(source.url) && (
                        <img
                            src={getFavicon(source.url)}
                            className="pill__fav"
                            alt=""
                            onError={(event) => {
                                event.currentTarget.style.display = "none";
                            }}
                        />
                    )}
                    <span className="pill__n">{index + 1}</span>
                    <span className="pill__domain">{getDomain(source.url)}</span>
                </a>
            ))}
        </div>
    );
}

function UserMsg({ text }) {
    return (
        <div className="umsg">
            <div className="umsg__av"><span>W</span></div>
            <div className="umsg__body">
                <div className="umsg__name">Winsley Saavedra</div>
                <div className="umsg__text">{text}</div>
            </div>
            <div className="umsg__acts">
                <button className="act-btn" title="Edit">✏</button>
                <button className="act-btn" title="Copy">⧉</button>
            </div>
        </div>
    );
}

function BotMsg({ msg, isLast, streaming }) {
    const { heading, body, sources = [], showPlanning, showSearching, queries = [], searchDone } = msg;
    const active = isLast && streaming;

    return (
        <div className="bmsg">
            {showPlanning && <PlanningCard done={searchDone} />}
            {showSearching && <SearchingCard queries={queries} done={searchDone} />}
            {(body || (active && !showSearching)) && (
                <div className="bmsg__ans">
                    {heading && <h2 className="bmsg__heading">{heading}</h2>}
                    <div className="bmsg__body">
                        <Markdown text={body || ""} sources={sources} />
                        {active && <span className="caret">▍</span>}
                    </div>
                    <SourcePills sources={sources} />
                </div>
            )}
        </div>
    );
}

function Landing({ onSearch }) {
    const [query, setQuery] = useState("");
    const inputRef = useRef(null);

    useEffect(() => { inputRef.current?.focus(); }, []);

    const submit = (event) => {
        event?.preventDefault();
        if (query.trim()) onSearch(query.trim());
    };

    const examples = [
        "How does quantum computing threaten modern encryption?",
        'site:arxiv.org "retrieval augmented generation" after:2024-01-01',
        "Best open source LLMs benchmark 2025",
        "intitle:CVE Apache Log4j critical vulnerability",
    ];

    return (
        <div className="land">
            <div className="land__logo"><span className="land__star">✳</span><span>nub-agent</span></div>
            <h1 className="land__h1">What do you want to know?</h1>
            <p className="land__sub">AI research with real citations, dork operators, and deep web reading.</p>
            <form className="land__form" onSubmit={submit}>
                <input
                    ref={inputRef}
                    className="land__in"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Ask anything or use site: filetype: intitle: operators..."
                    autoComplete="off"
                />
                <button type="submit" className="land__btn" disabled={!query.trim()}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2L14 8L8 14M2 8H14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </button>
            </form>
            <div className="land__exs">
                {examples.map((example) => <button key={example} className="land__ex" onClick={() => onSearch(example)}>{example}</button>)}
            </div>
        </div>
    );
}

export default function SearchEngine() {
    const [sessions, setSessions] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [input, setInput] = useState("");
    const [navActive, setNavActive] = useState("search");

    const abortRef = useRef(null);
    const bottomRef = useRef(null);
    const inputRef = useRef(null);

    const active = sessions.find((session) => session.id === activeId);
    const isLanding = !active && !streaming;

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [sessions, streaming]);

    const patchLastBot = useCallback((sessionId, patch) => {
        setSessions((previous) => previous.map((session) => {
            if (session.id !== sessionId) return session;
            const messages = [...session.messages];
            let index = -1;
            for (let i = messages.length - 1; i >= 0; i -= 1) {
                if (messages[i].role === "bot") {
                    index = i;
                    break;
                }
            }
            if (index < 0) return session;
            messages[index] = {
                ...messages[index],
                ...(typeof patch === "function" ? patch(messages[index]) : patch),
            };
            return { ...session, messages };
        }));
    }, []);

    const revealAnswer = useCallback(async (sessionId, finalText, sources, signal) => {
        const { heading, body } = extractAnswerParts(finalText);
        patchLastBot(sessionId, {
            heading,
            body: "",
            sources,
            showPlanning: false,
            showSearching: false,
            searchDone: true,
        });

        const chunks = body ? body.match(/.{1,34}(\s|$)/g) || [body] : [];
        let built = "";
        for (const chunk of chunks) {
            if (signal?.aborted) break;
            built += chunk;
            patchLastBot(sessionId, {
                heading,
                body: built.trimEnd(),
                sources,
                showPlanning: false,
                showSearching: false,
                searchDone: true,
            });
            await new Promise((resolve) => setTimeout(resolve, 18));
        }

        patchLastBot(sessionId, {
            heading,
            body: body || heading,
            sources,
            showPlanning: false,
            showSearching: false,
            searchDone: true,
        });
    }, [patchLastBot]);

    const runSearch = useCallback(async (query) => {
        if (!query.trim() || streaming) return;

        const sessionId = createId();
        const complex = query.length > 35 || /site:|filetype:|intitle:|inurl:|after:|before:/.test(query);
        const seedQueries = [
            query,
            ...query.trim().split(/\s+/).slice(0, 3).map((word, index) => (
                [`${word} definition`, `${word} software`, `what is ${word}`, `${word} meaning`][index] || word
            )),
        ].slice(0, 4);

        const newSession = {
            id: sessionId,
            query,
            messages: [
                { id: createId(), role: "user", text: query },
                {
                    id: createId(),
                    role: "bot",
                    heading: "",
                    body: "",
                    sources: [],
                    showPlanning: complex,
                    showSearching: true,
                    queries: seedQueries,
                    searchDone: false,
                },
            ],
        };

        setSessions((previous) => [...previous, newSession]);
        setActiveId(sessionId);
        setStreaming(true);
        setInput("");

        abortRef.current = new AbortController();
        const { signal } = abortRef.current;

        try {
            const history = active?.messages
                .filter((message) => message.role === "user" || (message.role === "bot" && message.body))
                .map((message) => (
                    message.role === "user"
                        ? { role: "user", content: message.text }
                        : { role: "assistant", content: `${message.heading ? `# ${message.heading}\n\n` : ""}${message.body}`.trim() }
                )) || [];

            const messages = [
                {
                    role: "system",
                    content: `You are a web research assistant. Always:
1. Search the web before answering any factual question.
2. Use dork operators for precision (site:, filetype:, intitle:, after:, etc).
3. Fetch top 2-3 URLs with web_fetch for full content.
4. Start your response with a clear title on the first line (use # Title).
5. Write inline citations [1][2][3] matching search result ranks when possible.
6. End with ## Sources listing each cited URL.
Never answer from memory alone - always search first.`,
                },
                ...history,
                { role: "user", content: query },
            ];

            const response = await fetch(CHAT_API, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: MODEL_NAME,
                    messages,
                    stream: false,
                }),
                credentials: "include",
                cache: "no-store",
                signal,
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const payload = await response.json();
            const fullText = payload?.choices?.[0]?.message?.content || payload?.output_text || "";
            const toolSources = extractToolSources(payload?.tools_used || payload?.toolsUsed || []);
            const answerSources = extractSources(fullText);
            const sources = [...toolSources];
            for (const source of answerSources) {
                if (!sources.find((entry) => entry.url === source.url)) sources.push(source);
            }

            patchLastBot(sessionId, { searchDone: true, showPlanning: false });
            await new Promise((resolve) => setTimeout(resolve, 250));
            await revealAnswer(sessionId, fullText, sources.slice(0, 8), signal);
        } catch (error) {
            if (error.name === "AbortError") {
                patchLastBot(sessionId, {
                    heading: "Stopped",
                    body: "Generation stopped.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
            } else {
                patchLastBot(sessionId, {
                    heading: "Error",
                    body: error.message || "Search failed. Try again.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
            }
        } finally {
            setStreaming(false);
            abortRef.current = null;
            setTimeout(() => inputRef.current?.focus(), 80);
        }
    }, [active, patchLastBot, revealAnswer, streaming]);

    const handleSubmit = (event) => {
        event?.preventDefault();
        if (input.trim()) runSearch(input.trim());
    };

    const NAV = [
        { id: "search", icon: "⊙", label: "Search" },
        { id: "library", icon: "⊟", label: "Library" },
        { id: "discover", icon: "◫", label: "Discover" },
        { id: "watcher", icon: "⊞", label: "Watcher" },
        { id: "finance", icon: "⊠", label: "Finance" },
    ];

    const NAV_BOTTOM = [
        { id: "pro", icon: "♛", label: "Pro" },
        { id: "settings", icon: "⚙", label: "Settings" },
        { id: "account", icon: "◉", label: "Account" },
    ];

    return (
        <div className="se">
            <style>{`
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c0f;--bg2:#111116;--bg3:#18181e;--bgc:#141418;
  --sur:rgba(255,255,255,0.04);--surh:rgba(255,255,255,0.07);
  --bdr:rgba(255,255,255,0.08);
  --tx:#ededf0;--txd:#8888a0;--txm:#484858;
  --ac:#00c9a7;--acd:rgba(0,201,167,0.14);--ac2:#6d6dff;
  --red:#ff4444;
  --r:12px;--sw:56px;
  --font:'DM Sans',system-ui,sans-serif;--mono:'DM Mono',monospace;
}
html,body,#root{height:100%;background:var(--bg)}
.se{display:flex;height:100vh;overflow:hidden;font-family:var(--font);color:var(--tx);background:var(--bg)}
.sb{width:var(--sw);flex:0 0 var(--sw);background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;align-items:center;padding:14px 0;gap:2px;z-index:20}
.sb__logo{width:34px;height:34px;border-radius:9px;background:var(--ac);display:flex;align-items:center;justify-content:center;font-size:17px;color:#000;margin-bottom:14px;cursor:pointer;box-shadow:0 0 18px rgba(0,201,167,.28)}
.sb__new{width:34px;height:34px;border-radius:9px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-size:17px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;margin-bottom:6px}
.sb__new:hover{background:var(--surh);color:var(--tx)}
.sb__sp{flex:1}
.nb{width:38px;height:38px;border-radius:9px;border:none;background:transparent;color:var(--txm);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;position:relative}
.nb:hover{background:var(--sur);color:var(--txd)}
.nb--on{background:var(--sur);color:var(--tx)}
.nb--on::before{content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);width:3px;height:18px;background:var(--ac);border-radius:0 2px 2px 0}
.mn{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
.tb{height:50px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;padding:0 22px;gap:12px;background:var(--bg2);flex:0 0 auto}
.tb__q{font-size:14px;font-weight:500;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tb__b{padding:5px 12px;border-radius:8px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-family:var(--font);font-size:12px;cursor:pointer;transition:all .2s}
.tb__b:hover{background:var(--surh);color:var(--tx)}
.tb__up{padding:5px 13px;border-radius:8px;border:none;background:var(--ac);color:#000;font-family:var(--font);font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:5px}
.chat{flex:1;overflow-y:auto;padding:28px 0 180px}
.chat::-webkit-scrollbar{width:4px}
.chat::-webkit-scrollbar-thumb{background:rgba(255,255,255,.07);border-radius:2px}
.chat__in{max-width:740px;margin:0 auto;padding:0 24px;display:flex;flex-direction:column;gap:24px}
.umsg{display:flex;align-items:flex-start;gap:11px}
.umsg__av{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#6d6dff,#00c9a7);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;flex:0 0 30px}
.umsg__body{flex:1;min-width:0}
.umsg__name{font-size:11px;font-weight:600;color:var(--txm);margin-bottom:3px}
.umsg__text{font-size:15px;color:var(--tx);line-height:1.6}
.umsg__acts{display:flex;gap:3px;opacity:0;transition:opacity .2s;padding-top:3px}
.umsg:hover .umsg__acts{opacity:1}
.act-btn{width:26px;height:26px;border-radius:6px;border:none;background:transparent;color:var(--txm);font-size:12px;cursor:pointer;transition:all .15s}
.act-btn:hover{background:var(--sur);color:var(--txd)}
.scard{background:var(--bgc);border:1px solid var(--bdr);border-radius:var(--r);padding:15px;display:flex;flex-direction:column;gap:11px;animation:fsi .28s ease;margin-left:41px}
.scard--done{opacity:.6}
@keyframes fsi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
.scard__head{display:flex;align-items:center;gap:9px}
.scard__icon{font-size:15px;flex:0 0 auto}
.scard__title{font-size:13px;font-weight:600}
.scard__sub{font-size:11px;color:var(--txm);margin-top:1px}
.scard__chips{display:flex;flex-wrap:wrap;gap:5px}
.qchip{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;background:var(--sur);border:1px solid var(--bdr);font-size:11px;color:var(--txd);animation:pi .22s ease both}
@keyframes pi{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}
.qchip__dot{font-size:9px;opacity:.5}
.scard__dots{display:flex;align-items:center;gap:3px}
.sdot{font-size:9px;color:var(--txm);transition:color .3s}
.sdot--on{color:var(--ac)}
.scard__searching{font-size:11px;color:var(--txd);margin-left:7px}
.scard__steps{display:flex;flex-direction:column;gap:7px}
.pstep{display:flex;align-items:flex-start;gap:7px;font-size:12px;transition:color .3s}
.pstep--wait{color:var(--txm)}
.pstep--done{color:var(--txd)}
.pstep__icon{font-size:11px;margin-top:1px;color:var(--ac);flex:0 0 auto;font-family:var(--mono)}
.pstep--wait .pstep__icon{color:var(--txm)}
.pstep__rest{color:var(--txm)}
.bmsg{display:flex;flex-direction:column;gap:11px;animation:fsi .3s ease}
.bmsg__ans{margin-left:41px;display:flex;flex-direction:column;gap:11px}
.bmsg__heading{font-size:21px;font-weight:700;letter-spacing:-.02em;line-height:1.2}
.bmsg__body{font-size:15px;line-height:1.78;color:var(--tx)}
.md-h1{font-size:19px;font-weight:700;margin:14px 0 5px;letter-spacing:-.02em}
.md-h2{font-size:16px;font-weight:600;margin:12px 0 4px}
.md-h3{font-size:14px;font-weight:600;margin:9px 0 3px;color:var(--txd)}
.md-p{margin:3px 0}
.md-ul{padding-left:18px;display:flex;flex-direction:column;gap:3px}
.ic{background:rgba(109,109,255,.12);border:1px solid rgba(109,109,255,.2);border-radius:4px;padding:1px 5px;font-family:var(--mono);font-size:12px;color:#a5a5ff}
.cite{display:inline-flex;align-items:center;justify-content:center;min-width:17px;height:17px;padding:0 3px;border-radius:4px;background:var(--acd);color:var(--ac);font-size:10px;font-weight:700;font-family:var(--mono);text-decoration:none;margin:0 2px;vertical-align:middle;border:1px solid rgba(0,201,167,.22)}
.cite:hover{background:rgba(0,201,167,.28)}
.caret{display:inline-block;color:var(--ac);animation:bl .65s step-end infinite;margin-left:2px}
@keyframes bl{0%,100%{opacity:1}50%{opacity:0}}
.pills{display:flex;flex-wrap:wrap;gap:5px;margin-top:3px}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;background:var(--sur);border:1px solid var(--bdr);font-size:11px;color:var(--txd);text-decoration:none;transition:all .2s}
.pill:hover{border-color:rgba(0,201,167,.3);color:var(--ac);background:var(--acd)}
.pill__fav{width:11px;height:11px;border-radius:2px}
.pill__n{font-size:9px;font-weight:700;font-family:var(--mono);color:var(--ac);background:var(--acd);border-radius:3px;padding:0 3px}
.pill__domain{max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bot{position:absolute;bottom:0;left:0;right:0;padding:14px 22px calc(14px + env(safe-area-inset-bottom));background:linear-gradient(transparent,var(--bg) 32%);z-index:10}
.bot__wrap{max-width:740px;margin:0 auto;background:var(--bg3);border:1px solid var(--bdr);border-radius:16px;overflow:hidden;transition:border-color .2s,box-shadow .2s}
.bot__wrap:focus-within{border-color:rgba(0,201,167,.35);box-shadow:0 0 0 3px rgba(0,201,167,.06)}
.bot__row{display:flex;align-items:center;padding:4px 4px 4px 16px;gap:7px}
.bot__in{flex:1;border:none;background:transparent;font-family:var(--font);font-size:15px;color:var(--tx);outline:none;padding:9px 0}
.bot__in::placeholder{color:var(--txm)}
.bot__in:disabled{opacity:.5}
.send-btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--ac);color:#000;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s;flex:0 0 auto}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.3;cursor:not-allowed}
.stop-btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--red);color:#fff;font-size:13px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex:0 0 auto;transition:opacity .2s}
.stop-btn:hover{opacity:.85}
.bot__bar{display:flex;align-items:center;padding:5px 9px;border-top:1px solid var(--bdr);gap:5px}
.bb{display:flex;align-items:center;gap:5px;padding:5px 10px;border-radius:7px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-family:var(--font);font-size:12px;cursor:pointer;transition:all .2s}
.bb:hover{background:var(--sur);color:var(--tx)}
.bb--on{background:var(--sur);color:var(--tx);border-color:rgba(255,255,255,.12)}
.bb__sp{flex:1}
.ib{width:30px;height:30px;border-radius:7px;border:none;background:transparent;color:var(--txm);font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s}
.ib:hover{background:var(--sur);color:var(--txd)}
.land{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;padding:40px 24px 190px;gap:18px;text-align:center}
.land__logo{display:flex;align-items:center;gap:7px;font-size:14px;font-weight:600;color:var(--txd);margin-bottom:6px}
.land__star{font-size:20px;color:var(--ac);filter:drop-shadow(0 0 10px var(--ac))}
.land__h1{font-size:clamp(24px,4vw,38px);font-weight:700;letter-spacing:-.03em;line-height:1.15}
.land__sub{font-size:14px;color:var(--txd);max-width:400px;line-height:1.65}
.land__form{width:100%;max-width:630px;background:var(--bg3);border:1px solid var(--bdr);border-radius:14px;display:flex;align-items:center;padding:4px 4px 4px 17px;gap:7px;margin-top:6px;transition:border-color .2s,box-shadow .2s}
.land__form:focus-within{border-color:rgba(0,201,167,.38);box-shadow:0 0 0 3px rgba(0,201,167,.06)}
.land__in{flex:1;border:none;background:transparent;font-family:var(--font);font-size:15px;color:var(--tx);outline:none;padding:10px 0}
.land__in::placeholder{color:var(--txm)}
.land__btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--ac);color:#000;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s;flex:0 0 auto}
.land__btn:disabled{opacity:.32;cursor:not-allowed}
.land__btn:hover:not(:disabled){opacity:.85}
.land__exs{display:flex;flex-direction:column;gap:5px;width:100%;max-width:630px;margin-top:2px}
.land__ex{padding:9px 15px;border-radius:9px;border:1px solid var(--bdr);background:var(--sur);color:var(--txd);font-family:var(--font);font-size:13px;cursor:pointer;transition:all .2s;text-align:left}
.land__ex:hover{border-color:rgba(0,201,167,.24);color:var(--tx);background:var(--acd)}
@media(max-width:600px){
  .tb{padding:0 14px}
  .chat__in{padding:0 14px}
  .bot{padding:10px 12px}
}
`}</style>

            <aside className="sb">
                <div className="sb__logo" onClick={() => { setActiveId(null); setInput(""); }}>✳</div>
                <button
                    className="sb__new"
                    onClick={() => {
                        setActiveId(null);
                        setInput("");
                        if (streaming) abortRef.current?.abort();
                    }}
                    title="New search"
                >
                    ＋
                </button>
                {NAV.map((item) => (
                    <button key={item.id} className={`nb ${navActive === item.id ? "nb--on" : ""}`} title={item.label} onClick={() => setNavActive(item.id)}>
                        {item.icon}
                    </button>
                ))}
                <div className="sb__sp" />
                {NAV_BOTTOM.map((item) => <button key={item.id} className="nb" title={item.label}>{item.icon}</button>)}
            </aside>

            <div className="mn">
                {!isLanding && (
                    <div className="tb">
                        <div className="tb__q">{active?.query || ""}</div>
                        <button className="tb__up">⬆ Upgrade to Pro</button>
                        <button className="tb__b">···</button>
                        <button className="tb__b">⬆ Share</button>
                    </div>
                )}

                <div className="chat">
                    {isLanding ? (
                        <Landing onSearch={runSearch} />
                    ) : (
                        <div className="chat__in">
                            {active?.messages.map((message, index) => (
                                message.role === "user"
                                    ? <UserMsg key={message.id} text={message.text} />
                                    : <BotMsg key={message.id} msg={message} isLast={index === active.messages.length - 1} streaming={streaming} />
                            ))}
                            <div ref={bottomRef} />
                        </div>
                    )}
                </div>

                {!isLanding && (
                    <div className="bot">
                        <div className="bot__wrap">
                            <form className="bot__row" onSubmit={handleSubmit}>
                                <input
                                    ref={inputRef}
                                    className="bot__in"
                                    value={input}
                                    onChange={(event) => setInput(event.target.value)}
                                    placeholder="Ask a new question..."
                                    disabled={streaming}
                                    autoComplete="off"
                                />
                                {streaming ? (
                                    <button type="button" className="stop-btn" onClick={() => abortRef.current?.abort()} title="Stop">■</button>
                                ) : (
                                    <button type="submit" className="send-btn" disabled={!input.trim()}>
                                        <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                                            <path d="M7.5 2L13 7.5L7.5 13M1 7.5H13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </button>
                                )}
                            </form>
                            <div className="bot__bar">
                                <button className="bb bb--on">🔍 Search</button>
                                <button className="bb">🔧</button>
                                <button className="bb">🔔</button>
                                <div className="bb__sp" />
                                <button className="ib" title="Attach">📎</button>
                                {streaming ? (
                                    <button className="ib" style={{ color: "var(--red)" }} onClick={() => abortRef.current?.abort()}>■</button>
                                ) : (
                                    <button className="ib" title="Voice">🎙</button>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
