const CANONICAL_MENU_TITLE = "Built with nubagent";
const CANONICAL_MENU_URL = "https://github.com/ambitiousnoob/nubagent";
const GET_STARTED_PAYLOAD = "NUBAGENT_GET_STARTED";
const MAX_PERSISTENT_MENU_ITEMS = 20;
const PROFILE_REPAIR_COOLDOWN_MS = 6 * 60 * 60 * 1000;

let sharedProfileRepairPromise = null;
let lastProfileRepairStartedAt = 0;

function summarizeProfileError(payload, status) {
  const message =
    payload?.error?.message ||
    payload?.error?.error_user_msg ||
    payload?.message ||
    "Unknown Messenger Profile API error";

  return `Messenger Profile API ${status}: ${message}`;
}

function buildProfileEndpoint(config) {
  return `https://graph.facebook.com/${config.graphApiVersion}/me/messenger_profile`;
}

function cloneMenuEntry(entry) {
  return {
    ...entry,
    call_to_actions: Array.isArray(entry?.call_to_actions)
      ? [...entry.call_to_actions]
      : [],
  };
}

function buildCanonicalMenuItem() {
  return {
    type: "web_url",
    title: CANONICAL_MENU_TITLE,
    url: CANONICAL_MENU_URL,
    webview_height_ratio: "full",
  };
}

function isCanonicalMenuItem(item) {
  return (
    item?.title === CANONICAL_MENU_TITLE || item?.url === CANONICAL_MENU_URL
  );
}

function mergeCanonicalMenuItem(callToActions, menuItem) {
  const existingItems = Array.isArray(callToActions)
    ? callToActions.filter((item) => !isCanonicalMenuItem(item))
    : [];
  let evictedItem = null;

  if (existingItems.length >= MAX_PERSISTENT_MENU_ITEMS) {
    evictedItem = existingItems.pop() || null;
  }

  existingItems.push(menuItem);

  return {
    callToActions: existingItems,
    evictedItem,
  };
}

function mergeDefaultLocaleMenu(persistentMenu, menuItem) {
  const existingMenu = Array.isArray(persistentMenu)
    ? persistentMenu.map(cloneMenuEntry)
    : [];
  let evictedItem = null;
  const defaultLocaleIndex = existingMenu.findIndex(
    (entry) => entry?.locale === "default",
  );

  if (defaultLocaleIndex >= 0) {
    const defaultLocaleEntry = existingMenu[defaultLocaleIndex];
    const updatedActions = mergeCanonicalMenuItem(
      defaultLocaleEntry?.call_to_actions,
      menuItem,
    );

    existingMenu[defaultLocaleIndex] = {
      ...defaultLocaleEntry,
      composer_input_disabled:
        defaultLocaleEntry?.composer_input_disabled === true,
      call_to_actions: updatedActions.callToActions,
    };
    evictedItem = updatedActions.evictedItem;
  } else {
    existingMenu.push({
      locale: "default",
      composer_input_disabled: false,
      call_to_actions: [menuItem],
    });
  }

  return {
    persistentMenu: existingMenu,
    evictedItem,
  };
}

async function requestProfile(config, { method, body, fields } = {}) {
  const url = new URL(buildProfileEndpoint(config));

  if (fields) {
    url.searchParams.set("fields", fields);
  }

  const response = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${config.pageAccessToken}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(summarizeProfileError(payload, response.status));
  }

  return payload;
}

function extractProfile(payload) {
  const data = Array.isArray(payload?.data) ? payload.data[0] : payload;
  return data && typeof data === "object" ? data : {};
}

function normalizePersistentMenu(persistentMenu) {
  return Array.isArray(persistentMenu)
    ? persistentMenu.map((entry) => ({
        locale: entry?.locale || "",
        composer_input_disabled: entry?.composer_input_disabled === true,
        call_to_actions: Array.isArray(entry?.call_to_actions)
          ? entry.call_to_actions.map((item) => ({
              type: item?.type || "",
              title: item?.title || "",
              url: item?.url || "",
              payload: item?.payload || "",
              webview_height_ratio: item?.webview_height_ratio || "",
            }))
          : [],
      }))
    : [];
}

function normalizeGetStarted(getStarted) {
  return {
    payload: getStarted?.payload || "",
  };
}

async function loadProfileState(config) {
  return extractProfile(
    await requestProfile(config, {
      method: "GET",
      fields: "persistent_menu,get_started",
    }),
  );
}

export async function ensureProfileState(config) {
  const profile = await loadProfileState(config);
  const menuItem = buildCanonicalMenuItem();
  const updatedMenu = mergeDefaultLocaleMenu(
    profile?.persistent_menu,
    menuItem,
  );
  const nextPersistentMenu = updatedMenu.persistentMenu;
  const currentPersistentMenu = normalizePersistentMenu(
    profile?.persistent_menu,
  );
  const desiredPersistentMenu = normalizePersistentMenu(nextPersistentMenu);
  const currentGetStarted = normalizeGetStarted(profile?.get_started);
  const needsGetStarted = !currentGetStarted.payload;
  const menuChanged =
    JSON.stringify(currentPersistentMenu) !==
    JSON.stringify(desiredPersistentMenu);
  const body = {};

  if (menuChanged) {
    body.persistent_menu = nextPersistentMenu;
  }

  if (needsGetStarted) {
    body.get_started = {
      payload: GET_STARTED_PAYLOAD,
    };
  }

  if (menuChanged || needsGetStarted) {
    await requestProfile(config, {
      method: "POST",
      body,
    });
  }

  return {
    changed: menuChanged || needsGetStarted,
    menuChanged,
    getStartedConfigured: needsGetStarted,
    defaultMenuItemCount:
      nextPersistentMenu.find((entry) => entry.locale === "default")
        ?.call_to_actions?.length || 0,
    evictedItem: updatedMenu.evictedItem,
    menuTitle: menuItem.title,
    menuUrl: menuItem.url,
  };
}

export function maybeRepairProfileState(config, logger = console) {
  if (!config?.pageAccessToken) {
    return null;
  }

  const now = Date.now();

  if (
    sharedProfileRepairPromise &&
    now - lastProfileRepairStartedAt < PROFILE_REPAIR_COOLDOWN_MS
  ) {
    return sharedProfileRepairPromise;
  }

  if (now - lastProfileRepairStartedAt < PROFILE_REPAIR_COOLDOWN_MS) {
    return null;
  }

  lastProfileRepairStartedAt = now;
  sharedProfileRepairPromise = ensureProfileState(config)
    .then((result) => {
      logger.info?.("Checked Messenger profile state", result);
      return result;
    })
    .catch((error) => {
      logger.warn?.("Failed to repair Messenger profile state", {
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    })
    .finally(() => {
      sharedProfileRepairPromise = null;
    });

  return sharedProfileRepairPromise;
}
