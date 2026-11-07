(() => {
  "use strict";

  // Dynamic API Base Resolution:
  // Supports custom injection via window.MINDSCOPE_API_BASE,
  // local development fallback (port 8000 or http://127.0.0.1:8000),
  // and production cloud deployment.
  const isLocal =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.protocol === "file:";

  const API_BASE =
    window.MINDSCOPE_API_BASE ||
    (isLocal
      ? (window.location.port === "8000" ? window.location.origin : "http://127.0.0.1:8000")
      : "https://mansik-santulan-score.onrender.com");

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const errorRetryBtn = document.getElementById("error-retry-btn");

  const stateIdle = document.getElementById("state-idle");
  const stateLoading = document.getElementById("state-loading");
  const stateResult = document.getElementById("state-result");
  const stateError = document.getElementById("state-error");

  const scoreNumberEl = document.getElementById("score-number");
  const scoreBandEl = document.getElementById("score-band");
  const scoreContextEl = document.getElementById("score-context");
  const gaugeFill = document.getElementById("gauge-fill");
  const errorLabelEl = document.getElementById("error-label");
  const errorCopyEl = document.getElementById("error-copy");

  const waterfallCard = document.getElementById("waterfall-card");
  const waterfallBadge = document.getElementById("waterfall-badge");
  const waterfallList = document.getElementById("waterfall-list");

  const GAUGE_ARC_LENGTH = 314; // approx pi * r(100)
  let isSubmitting = false;

  // ---------------------------------------------------------
  // Draw tick marks on both gauges (0..10, every 2 units)
  // ---------------------------------------------------------
  function drawTicks() {
    document.querySelectorAll(".gauge-ticks").forEach((g) => {
      g.innerHTML = "";
      const cx = 120, cy = 140, rOuter = 100, rInner = 90;
      for (let i = 0; i <= 10; i += 2) {
        const angle = Math.PI - (i / 10) * Math.PI; // 180deg -> 0deg
        const x1 = cx + rOuter * Math.cos(angle);
        const y1 = cy - rOuter * Math.sin(angle);
        const x2 = cx + rInner * Math.cos(angle);
        const y2 = cy - rInner * Math.sin(angle);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x1.toFixed(1));
        line.setAttribute("y1", y1.toFixed(1));
        line.setAttribute("x2", x2.toFixed(1));
        line.setAttribute("y2", y2.toFixed(1));
        g.appendChild(line);
      }
    });
  }
  drawTicks();

  // ---------------------------------------------------------
  // Segmented control (stress_level) wiring
  // ---------------------------------------------------------
  const segGroup = document.getElementById("stress_level_group");
  const stressHiddenInput = document.getElementById("stress_level");
  if (segGroup && stressHiddenInput) {
    segGroup.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        segGroup.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        stressHiddenInput.value = btn.dataset.value;
        clearFieldError(stressHiddenInput);
      });
    });
  }

  // ---------------------------------------------------------
  // Field-level error helpers
  // ---------------------------------------------------------
  function fieldWrapper(input) {
    return input ? input.closest(".field") : null;
  }

  function setFieldError(input, message) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.add("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = message;
  }

  function clearFieldError(input) {
    const wrap = fieldWrapper(input);
    if (!wrap) return;
    wrap.classList.remove("field-error");
    const msgEl = wrap.querySelector(".error-msg");
    if (msgEl) msgEl.textContent = "";
  }

  function clearAllErrors() {
    if (!form) return;
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    form.querySelectorAll(".error-msg").forEach((m) => (m.textContent = ""));
  }

  // ---------------------------------------------------------
  // Client-side validation mirroring the StudentData model
  // ---------------------------------------------------------
  function validate(payload) {
    const errors = [];

    const numericChecks = [
      ["age", 18, 24],
      ["avg_daily_usage_hours", 0, 16],
      ["daily_unlocks", 0, 500],
      ["study_hours", 0, 16],
      ["physical_activity_hours", 0, 8],
      ["sleep_hours_per_night", 2, 16],
    ];

    numericChecks.forEach(([key, min, max]) => {
      const input = document.getElementById(key);
      const val = payload[key];
      if (val === "" || val === null || Number.isNaN(val)) {
        errors.push([input, "This field is required."]);
      } else if (val < min || val > max) {
        errors.push([input, `Must be between ${min} and ${max === Infinity ? "0+" : max}.`]);
      }
    });

    ["gender", "country", "academic_level", "most_used_platform", "purpose_of_use"].forEach((key) => {
      const input = document.getElementById(key);
      if (!payload[key] || String(payload[key]).trim() === "") {
        errors.push([input, "This field is required."]);
      }
    });

    if (!payload.stress_level) {
      errors.push([stressHiddenInput, "Pick a stress level."]);
    }

    return errors;
  }

  // ---------------------------------------------------------
  // Gather form data into the exact StudentData shape
  // ---------------------------------------------------------
  function collectPayload() {
    if (!form) return {};
    const fd = new FormData(form);
    return {
      age: fd.get("age") === "" ? NaN : parseInt(fd.get("age"), 10),
      gender: fd.get("gender") || "",
      country: (fd.get("country") || "").trim(),
      academic_level: fd.get("academic_level") || "",
      most_used_platform: fd.get("most_used_platform") || "",
      purpose_of_use: fd.get("purpose_of_use") || "",
      avg_daily_usage_hours: fd.get("avg_daily_usage_hours") === "" ? NaN : parseFloat(fd.get("avg_daily_usage_hours")),
      daily_unlocks: fd.get("daily_unlocks") === "" ? NaN : parseInt(fd.get("daily_unlocks"), 10),
      study_hours: fd.get("study_hours") === "" ? NaN : parseFloat(fd.get("study_hours")),
      physical_activity_hours: fd.get("physical_activity_hours") === "" ? NaN : parseFloat(fd.get("physical_activity_hours")),
      sleep_hours_per_night: fd.get("sleep_hours_per_night") === "" ? NaN : parseFloat(fd.get("sleep_hours_per_night")),
      stress_level: fd.get("stress_level") || "",
    };
  }

  // ---------------------------------------------------------
  // UI state switching
  // ---------------------------------------------------------
  function showState(name) {
    const states = { idle: stateIdle, loading: stateLoading, result: stateResult, error: stateError };
    [stateIdle, stateLoading, stateResult, stateError].forEach((el) => {
      if (el) el.hidden = true;
    });
    if (states[name]) {
      states[name].hidden = false;
    }
  }

  function setSubmitting(submitting) {
    isSubmitting = submitting;
    if (submitBtn) {
      submitBtn.disabled = submitting;
      submitBtn.classList.toggle("loading", submitting);
    }
  }

  function bandFor(score) {
    if (score < 4) {
      return {
        label: "Signal: strained",
        context: "Your responses suggest elevated strain right now. Small shifts in sleep or screen time can go a long way.",
      };
    }
    if (score < 7) {
      return {
        label: "Signal: balanced",
        context: "Your rhythm looks fairly steady, with some room to recover and reset.",
      };
    }
    return {
      label: "Signal: strong",
      context: "Your habits point to a well-supported, resilient baseline. Keep it up.",
    };
  }

  function renderWaterfall(explanation) {
    if (!waterfallCard || !waterfallList) return;
    if (!explanation || !Array.isArray(explanation.feature_attributions)) {
      waterfallCard.hidden = true;
      return;
    }

    if (waterfallBadge && typeof explanation.expected_baseline_score === "number") {
      waterfallBadge.textContent = `Expected Model Baseline: ${explanation.expected_baseline_score.toFixed(2)}`;
    }

    waterfallList.innerHTML = "";
    explanation.feature_attributions.forEach((attr) => {
      const row = document.createElement("div");
      row.className = "waterfall-row";

      const left = document.createElement("div");
      left.className = "waterfall-row-left";

      const nameEl = document.createElement("span");
      nameEl.className = "waterfall-feat-name";
      nameEl.textContent = attr.display_name || attr.feature;

      const valEl = document.createElement("span");
      valEl.className = "waterfall-feat-val";
      valEl.textContent = attr.formatted_value || String(attr.value);

      left.appendChild(nameEl);
      left.appendChild(valEl);

      const right = document.createElement("div");
      right.className = "waterfall-row-right";

      const pill = document.createElement("span");
      const isPos = attr.shap_value >= 0;
      pill.className = `waterfall-pill ${isPos ? "pill-positive" : "pill-negative"}`;
      const arrow = isPos ? "↑ +" : "↓ −";
      const absVal = Math.abs(attr.shap_value).toFixed(2);
      pill.textContent = `${arrow}${absVal}`;
      pill.title = isPos
        ? "Moves the model prediction above its expected baseline."
        : "Moves the model prediction below its expected baseline.";

      right.appendChild(pill);
      row.appendChild(left);
      row.appendChild(right);
      waterfallList.appendChild(row);
    });

    waterfallCard.hidden = false;
  }

  function renderResult(score, explanation) {
    const clamped = Math.max(0, Math.min(10, score));
    const { label, context } = bandFor(clamped);

    if (scoreNumberEl) scoreNumberEl.textContent = score.toFixed(2);
    if (scoreBandEl) scoreBandEl.textContent = label;
    if (scoreContextEl) scoreContextEl.textContent = context;

    if (gaugeFill) {
      // reset then animate the arc fill on next frame
      gaugeFill.style.transition = "none";
      gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
      requestAnimationFrame(() => {
        gaugeFill.style.transition = "";
        const offset = GAUGE_ARC_LENGTH * (1 - clamped / 10);
        gaugeFill.style.strokeDashoffset = String(offset);
      });
    }

    renderWaterfall(explanation);
    showState("result");
  }

  function renderError(label, copy) {
    if (errorLabelEl) errorLabelEl.textContent = label;
    if (errorCopyEl) errorCopyEl.textContent = copy;
    showState("error");
  }

  // ---------------------------------------------------------
  // Parse FastAPI / Pydantic 422 error responses into
  // field-level messages where possible
  // ---------------------------------------------------------
  function applyServerValidationErrors(detail) {
    if (!Array.isArray(detail)) return false;
    let matched = false;
    detail.forEach((err) => {
      const field = Array.isArray(err.loc) ? err.loc[err.loc.length - 1] : null;
      const input = field ? document.getElementById(field) : null;
      const target = field === "stress_level" ? stressHiddenInput : input;
      if (target) {
        setFieldError(target, err.msg || "Invalid value.");
        matched = true;
      }
    });
    return matched;
  }

  // ---------------------------------------------------------
  // Submit handler with concurrency lock & resilient error handling
  // ---------------------------------------------------------
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      // Prevent concurrent duplicate submissions
      if (isSubmitting) return;

      clearAllErrors();

      const payload = collectPayload();
      const clientErrors = validate(payload);

      if (clientErrors.length > 0) {
        clientErrors.forEach(([input, msg]) => input && setFieldError(input, msg));
        clientErrors[0][0]?.focus?.();
        return;
      }

      setSubmitting(true);
      showState("loading");

      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        let body = null;
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          try {
            body = await res.json();
          } catch {
            body = null;
          }
        }

        if (res.status === 422) {
          const matched = body && applyServerValidationErrors(body.detail);
          renderError(
            "Check your inputs",
            matched
              ? "The API rejected a few fields — details are marked on the form."
              : "The API rejected this submission. Please review your inputs and try again."
          );
          return;
        }

        if (!res.ok) {
          let detailMsg = `The API responded with status ${res.status}.`;
          if (body && typeof body.detail === "string") {
            detailMsg = body.detail;
          } else if (body && typeof body.message === "string") {
            detailMsg = body.message;
          }
          renderError("Prediction failed", detailMsg);
          return;
        }

        if (!body || typeof body.predicted_mental_health_score !== "number") {
          renderError("Unexpected response", "The API responded, but the score was missing or malformed.");
          return;
        }

        // Fetch local XAI explanation asynchronously
        let explanationData = null;
        try {
          const expRes = await fetch(`${API_BASE}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (expRes.ok) {
            explanationData = await expRes.json();
          }
        } catch {
          explanationData = null;
        }

        renderResult(body.predicted_mental_health_score, explanationData);
      } catch (err) {
        renderError(
          "Can't reach the server",
          `Couldn't connect to ${API_BASE}. Make sure the backend is running (uvicorn main:app --port 8000 --reload) and reachable from this page.`
        );
      } finally {
        setSubmitting(false);
      }
    });

    // live-clear errors as the user edits
    form.querySelectorAll("input, select").forEach((el) => {
      el.addEventListener("input", () => clearFieldError(el));
      el.addEventListener("change", () => clearFieldError(el));
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (waterfallCard) waterfallCard.hidden = true;
      showState("idle");
    });
  }

  if (errorRetryBtn) {
    errorRetryBtn.addEventListener("click", () => {
      if (waterfallCard) waterfallCard.hidden = true;
      showState("idle");
    });
  }
})();
