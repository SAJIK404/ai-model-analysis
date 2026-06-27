const DATA_URL = "data.json";

function createTable(headers, rows) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((header) => {
      const key = header.toLowerCase().replace(/ /g, "_");
      const td = document.createElement("td");
      td.textContent = row[key] ?? "-";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function renderCharts(charts) {
  const chartGrid = document.getElementById("chart-grid");
  chartGrid.innerHTML = "";
  charts.forEach((chart) => {
    const card = document.createElement("div");
    card.className = "chart-card";

    const img = document.createElement("img");
    img.src = chart.file;
    img.alt = chart.title;
    card.appendChild(img);

    const body = document.createElement("div");
    body.className = "chart-card-body";

    const title = document.createElement("h3");
    title.textContent = chart.title;
    body.appendChild(title);

    const caption = document.createElement("p");
    caption.textContent = "Haz clic en la imagen para abrirla en una nueva pestaña.";
    body.appendChild(caption);

    card.appendChild(body);
    chartGrid.appendChild(card);

    card.addEventListener("click", () => {
      window.open(chart.file, "_blank");
    });
  });
}

function renderTable(sectionId, headers, rows) {
  const container = document.getElementById(sectionId);
  container.innerHTML = "";
  container.appendChild(createTable(headers, rows));
}

function formatDate(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  return date.toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short" });
}

async function loadSiteData() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`No se pudo cargar ${DATA_URL}`);
  }
  return await response.json();
}

async function init() {
  try {
    const data = await loadSiteData();
    document.getElementById("last-run").textContent = formatDate(data.project.last_run);
    document.getElementById("commit-hash").textContent = data.project.commit || "No disponible";

    renderCharts(data.charts);

    renderTable("value-score-table", [
      "model_id",
      "provider",
      "model_name",
      "avg_benchmark_score",
      "avg_cost_per_1m",
      "value_score",
    ], data.metrics.value_score_ranking);

    renderTable("best-tier-table", [
      "price_tier",
      "model_id",
      "provider",
      "model_name",
      "avg_benchmark_score",
      "avg_cost_per_1m",
      "value_score",
    ], data.metrics.best_per_tier);

    renderTable("category-table", [
      "category",
      "model_id",
      "provider",
      "model_name",
      "category_score",
      "category_rank",
    ], data.metrics.category_rankings);

    renderTable("benchmark-table", [
      "benchmark_name",
      "model_id",
      "provider",
      "model_name",
      "score",
      "benchmark_rank",
    ], data.metrics.top_model_per_benchmark);
  } catch (error) {
    const root = document.querySelector("main");
    root.innerHTML = `<div class="error"><h2>Error al cargar datos</h2><p>${error.message}</p></div>`;
    console.error(error);
  }
}

window.addEventListener("DOMContentLoaded", init);
