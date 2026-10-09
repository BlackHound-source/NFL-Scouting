let DATA = { meta: { mode_message: "", score_label: "" }, players: [] };

async function loadData() {
  try {
    const res = await fetch('/d2tf_scout_payload.json');
    if (res.ok) {
      DATA = await res.json();
    }
  } catch (err) {
    console.error("Failed to load payload:", err);
  }
  initApp();
}

function initApp() {
  document.getElementById('banner').textContent = DATA.meta.mode_message || "D2-TF Operational Mode Active";
  
  // Setup position filters dynamically
  const positions = [...new Set(DATA.players.map(p => p.position))].sort();
  const posContainer = document.getElementById('posf');
  posContainer.innerHTML = positions.map(pos => 
    `<label style="display:block; margin-bottom:3px;"><input type="checkbox" class="pos-filter" value="${pos}" checked> ${pos}</label>`
  ).join('');

  document.querySelectorAll('.pos-filter').forEach(cb => cb.addEventListener('change', renderCards));
  document.getElementById('minscore').addEventListener('input', renderCards);
  document.getElementById('q').addEventListener('input', renderCards);

  renderCards();
}

function renderCards() {
  const selectedPositions = [...document.querySelectorAll('.pos-filter:checked')].map(cb => cb.value);
  const minScore = parseFloat(document.getElementById('minscore').value) || 0;
  const searchQuery = document.getElementById('q').value.toLowerCase().trim();

  const filtered = DATA.players.filter(p => {
    if (!selectedPositions.includes(p.position)) return false;
    if (p.pred_rr < minScore) return false;
    if (searchQuery && !p.name.toLowerCase().includes(searchQuery)) return false;
    return true;
  }).sort((a, b) => b.pred_rr - a.pred_rr);

  const grid = document.getElementById('cardGrid');
  if (filtered.length === 0) {
    grid.innerHTML = '<p style="color:#888;">No players match the current filters.</p>';
    return;
  }

  grid.innerHTML = filtered.map(player => {
    // Map D2-TF score (0-1) to FIFA rating (50-99)
    const rating = Math.min(99, Math.max(50, Math.round(player.pred_rr * 49 + 50)));
    const rec = Math.round((player.duel_metrics['shrunk recovery rate'] || 0.5) * 100);
    const dev = Math.round((player.duel_metrics['deficit elimination velocity'] || 0.5) * 100);
    const cov = Math.round(player.coverage * 100);
    const rank = player.pos_rank || 1;

    return `
      <div class="fifa-card" onclick="alert('${player.name}\\nPosition Rank: #${rank}\\nScore: ${player.pred_rr.toFixed(3)}')">
        <div class="card-header">
          <span class="card-rating">${rating}</span>
          <span class="card-position">${player.position}</span>
        </div>
        <div class="card-body">
          <div class="player-name">${player.name}</div>
          <div class="attributes-grid">
            <div class="attr-item"><span>${rec}</span> REC</div>
            <div class="attr-item"><span>${dev}</span> DEV</div>
            <div class="attr-item"><span>${cov}</span> COV</div>
            <div class="attr-item"><span>#${rank}</span> POS</div>
          </div>
        </div>
        <div class="card-footer">${player.kind}</div>
      </div>
    `;
  }).join('');
}

loadData();
