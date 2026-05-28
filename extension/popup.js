// ─────────────────────────────────────────────────────────────
// SMART JOB EXTRACTOR
// Targets only the open job panel, not the entire page
// ─────────────────────────────────────────────────────────────

function extractJobText() {
    const host = window.location.hostname;

    // ── Indeed ────────────────────────────────────────────────
    if (host.includes("indeed.com")) {
        const selectors = [
            '[data-testid="jobsearch-JobComponent"]',
            '.jobsearch-JobComponent',
            '#jobDescriptionText',
            '.job_seen_beacon:first-child',
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > 100) return el.innerText.trim();
        }
    }

    // ── LinkedIn ──────────────────────────────────────────────
    if (host.includes("linkedin.com")) {
        let parts = [];
        const selectors = [
            '.jobs-search__job-details',
            '.job-view-layout',
            '.jobs-unified-top-card',
            '.jobs-description',
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) parts.push(el.innerText.trim());
        }
        if (parts.length) return parts.join(" ");
    }

    // ── Naukri ────────────────────────────────────────────────
    if (host.includes("naukri.com")) {
        const selectors = ['.job-desc', '.jd-container', '#job_header'];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > 100) return el.innerText.trim();
        }
    }

    // ── Internshala ───────────────────────────────────────────
    if (host.includes("internshala.com")) {
        const selectors = ['.internship_details', '#internship_detail_container'];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > 50) return el.innerText.trim();
        }
    }

    // ── Glassdoor ─────────────────────────────────────────────
    if (host.includes("glassdoor.com")) {
        const selectors = [
            '[data-test="jobDescriptionContent"]',
            '.jobDescriptionContent',
            '.desc',
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > 100) return el.innerText.trim();
        }
    }

    // ── Generic fallback: biggest non-nav block ───────────────
    const candidates = document.querySelectorAll(
        "article, main, section, [class*='job'], [class*='desc'], [id*='job'], [id*='desc']"
    );
    let best = null, bestLen = 0;
    for (const el of candidates) {
        const cls = (el.className + " " + el.id).toLowerCase();
        if (/nav|sidebar|header|footer|list|search|filter|cookie|banner/.test(cls)) continue;
        const len = el.innerText.trim().length;
        if (len > bestLen && len < 20000) { bestLen = len; best = el; }
    }
    if (best && bestLen > 200) return best.innerText.trim();

    // Last resort — capped at 6000 chars
    return document.body.innerText.trim().slice(0, 6000);
}

// ─────────────────────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────────────────────

function updateUI(data) {
    const resultDiv = document.getElementById('result');

    if (data.status === "error") {
        resultDiv.textContent = `❌ Error: ${data.message}`;
        resultDiv.className = "scam";
        return;
    }

    // Build flag list HTML (shown for BOTH safe and scam results)
    let flagHtml = "";
    if (data.flags && data.flags.length > 0) {
        const items = data.flags.map(f => `<li>"${f}"</li>`).join('');
        flagHtml = `
            <div style="text-align:left;margin-top:12px;padding:10px;
                        background:#fff3f3;border-radius:6px;border:1px solid #ffcccc;">
                <strong style="font-size:13px;color:#cc0000;">🚩 Red Flags Detected:</strong>
                <ul style="margin:5px 0 0 0;padding-left:20px;font-size:13px;
                           font-weight:normal;color:#333;">${items}</ul>
            </div>`;
    }

    if (data.is_fake) {
        resultDiv.innerHTML = `🔴 <strong>Scam Probability: ${data.scam_probability}%</strong>
            <br><span style="font-size:12px;font-weight:normal">Flagged as likely fraudulent</span>
            ${flagHtml}`;
        resultDiv.className = "scam";
    } else {
        resultDiv.innerHTML = `🟢 <strong>Safe. Scam Probability: ${data.scam_probability}%</strong>
            <br><span style="font-size:12px;font-weight:normal">Looks legitimate</span>
            ${flagHtml}`;
        resultDiv.className = "safe";
    }
}

function setLoading(msg = "Scanning job...") {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = msg;
    resultDiv.className = "loading";
}

// ─────────────────────────────────────────────────────────────
// 1. AUTO-SCAN (runs immediately when popup opens)
// ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    setLoading("Auto-scanning job...");

    try {
        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        let injectionResults = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: extractJobText,
        });

        const jobText = injectionResults[0].result;
        console.log(`[FJD] Extracted ${jobText?.length ?? 0} chars`);

        if (!jobText || jobText.length < 50) {
            document.getElementById('result').textContent = "No job content found. Try pasting the link below.";
            document.getElementById('result').className = "loading";
            return;
        }

        const response = await fetch('https://fake-job-detector-production-31b1.up.railway.app/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: jobText })
        });

        const data = await response.json();
        updateUI(data);

    } catch (error) {
        console.error("[FJD] Auto-scan error:", error);
        document.getElementById('result').textContent = "Could not scan page — try the link scanner below.";
        document.getElementById('result').className = "loading";
    }
});

// ─────────────────────────────────────────────────────────────
// 2. FILE UPLOAD HANDLER
// ─────────────────────────────────────────────────────────────

document.getElementById('file-upload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading("Analyzing file...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch('https://fake-job-detector-production-31b1.up.railway.app/predict_file', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        document.getElementById('result').textContent = "Server error. Make sure FastAPI is running!";
        document.getElementById('result').className = "scam";
    }
});

// ─────────────────────────────────────────────────────────────
// 3. URL LINK SCANNER
// ─────────────────────────────────────────────────────────────

document.getElementById('url-btn').addEventListener('click', async () => {
    const urlValue = document.getElementById('url-input').value.trim();
    if (!urlValue) return;

    setLoading("Scraping link...");

    try {
        const response = await fetch('https://fake-job-detector-production-31b1.up.railway.app/predict_url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlValue })
        });
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        document.getElementById('result').textContent = "Server error. Make sure FastAPI is running!";
        document.getElementById('result').className = "scam";
    }
});
