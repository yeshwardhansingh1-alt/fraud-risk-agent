let scenarios = [];
let cumPrevented = 0;
let cumFpCost = 0;

document.addEventListener("DOMContentLoaded", async () => {
  // Load scenarios
  try {
    const res = await fetch("/demo/scenarios");
    scenarios = await res.json();
    renderScenarios();
  } catch (e) {
    console.error("Failed to load scenarios:", e);
  }

  // Handle form submission
  document.getElementById("payment-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await processTransaction();
  });
});

function renderScenarios() {
  const container = document.getElementById("scenarios-container");
  scenarios.forEach((s) => {
    const btn = document.createElement("button");
    btn.className = "scenario-btn";
    btn.type = "button";
    btn.innerText = s.name;
    btn.title = s.description;
    btn.onclick = () => loadScenario(s);
    container.appendChild(btn);
  });
}

function loadScenario(s) {
  const p = s.params;
  document.getElementById("f-amount").value = p.amount;
  document.getElementById("f-card").value = p.card_type;
  document.getElementById("f-email").value = p.email_domain;
  document.getElementById("f-device").value = p.device_type;
  document.getElementById("f-hour").value = p.hour;
  document.getElementById("f-repeat").checked = p.is_repeat_customer;
  document.getElementById("f-v10m").value = p.velocity_10min;
  document.getElementById("f-v1h").value = p.velocity_1hr;
  document.getElementById("f-v24h").value = p.velocity_24hr;
  document.getElementById("f-cards-device").value = p.cards_on_device;
  document.getElementById("f-travel").checked = p.impossible_travel;
}

async function processTransaction() {
  const btn = document.getElementById("btn-process");
  btn.innerText = "⏳ Processing...";
  btn.disabled = true;

  const payload = {
    amount: parseFloat(document.getElementById("f-amount").value),
    card_type: document.getElementById("f-card").value,
    email_domain: document.getElementById("f-email").value,
    device_type: document.getElementById("f-device").value,
    hour: parseInt(document.getElementById("f-hour").value),
    is_repeat_customer: document.getElementById("f-repeat").checked,
    velocity_10min: parseInt(document.getElementById("f-v10m").value),
    velocity_1hr: parseInt(document.getElementById("f-v1h").value),
    velocity_24hr: parseInt(document.getElementById("f-v24h").value),
    cards_on_device: parseInt(document.getElementById("f-cards-device").value),
    amount_zscore: 0.0, // simplified
    impossible_travel: document.getElementById("f-travel").checked
  };

  try {
    const res = await fetch("/predict_demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) throw new Error("API Error");
    
    const data = await res.json();
    renderDecision(data, payload.amount);
    addToHistory(data, payload.amount);
    updateCounters(data, payload.amount);
    
  } catch (e) {
    console.error(e);
    alert("Error processing transaction. Make sure the API is running.");
  } finally {
    btn.innerText = "💳 Process Payment";
    btn.disabled = false;
  }
}

function renderDecision(data, amount) {
  const panel = document.getElementById("decision-panel");
  panel.classList.add("active");

  // Action badge
  const actionDiv = document.getElementById("d-action");
  actionDiv.className = `action-badge action-${data.action}`;
  actionDiv.innerHTML = `${data.action_emoji} ${data.action.replace('_', ' ').toUpperCase()}`;

  // Probability
  document.getElementById("d-prob").innerText = `${(data.fraud_probability * 100).toFixed(2)}%`;
  document.getElementById("d-risk").innerText = `Risk Level: ${data.risk_level}`;
  
  const fill = document.getElementById("d-prob-bar");
  fill.style.width = `${Math.min(data.fraud_probability * 100, 100)}%`;

  // Reason
  document.getElementById("d-reason").innerText = data.reason;

  // Expected Losses Bar Chart
  const lossChart = document.getElementById("d-loss-chart");
  lossChart.innerHTML = "";
  
  // Find max loss to scale the bars
  let maxLoss = 0;
  for (const v of Object.values(data.expected_losses)) {
    if (v > maxLoss) maxLoss = v;
  }
  if (maxLoss === 0) maxLoss = 1;

  for (const [action, loss] of Object.entries(data.expected_losses)) {
    const w = (loss / maxLoss) * 100;
    const isChosen = action === data.action;
    
    let color = "#ccc";
    if (action === "approve") color = "var(--color-approve)";
    if (action === "step_up") color = "var(--color-stepup)";
    if (action === "block") color = "var(--color-block)";
    
    lossChart.innerHTML += `
      <div class="loss-row">
        <div class="loss-label" style="${isChosen ? 'font-weight:bold;color:#fff;' : ''}">${action.replace('_', ' ')}</div>
        <div class="loss-bar-wrap">
          <div class="loss-bar" style="width: ${w}%; background: ${color}; opacity: ${isChosen ? 1 : 0.4}"></div>
        </div>
        <div class="loss-val" style="${isChosen ? 'font-weight:bold;color:#fff;' : ''}">$${loss.toFixed(2)}</div>
      </div>
    `;
  }

  // SHAP
  const shapList = document.getElementById("d-shap-list");
  shapList.innerHTML = "";
  data.top_features.forEach(f => {
    const isUp = f.shap_value > 0;
    const arrow = isUp ? "↑" : "↓";
    const cssClass = isUp ? "shap-increases" : "shap-decreases";
    shapList.innerHTML += `
      <li class="${cssClass}">
        <span>${arrow} ${f.feature}</span>
        <span>${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(2)}</span>
      </li>
    `;
  });
}

function addToHistory(data, amount) {
  const container = document.getElementById("history-container");
  
  const el = document.createElement("div");
  el.className = `history-item ${data.action}`;
  el.innerHTML = `
    <div>
      <strong>${data.transaction_id}</strong> &mdash; $${amount.toFixed(2)}
      <div style="font-size: 0.8rem; color: #aaa; margin-top:0.2rem;">Prob: ${(data.fraud_probability * 100).toFixed(1)}%</div>
    </div>
    <div style="text-align: right;">
      <span style="font-weight:bold;">${data.action_emoji} ${data.action.replace('_', ' ').toUpperCase()}</span>
      <div style="font-size: 0.8rem; color: #aaa; margin-top:0.2rem;">${data.risk_level} Risk</div>
    </div>
  `;
  
  container.prepend(el);
}

function updateCounters(data, amount) {
  // Simplified hypothetical updating logic for demo engagement
  // If we blocked a high risk, assume we saved it.
  if (data.action === "block" && data.fraud_probability > 0.5) {
    cumPrevented += amount;
  }
  // If we stepped up, assume some friction cost
  if (data.action === "step_up") {
    cumFpCost += 5.00; 
  }
  // If we blocked a low risk, assume we lost it as FP
  if (data.action === "block" && data.fraud_probability < 0.2) {
    cumFpCost += amount * 0.8;
  }

  document.getElementById("stat-prevented").innerText = `$${cumPrevented.toFixed(2)}`;
  document.getElementById("stat-fpcost").innerText = `$${cumFpCost.toFixed(2)}`;
}
