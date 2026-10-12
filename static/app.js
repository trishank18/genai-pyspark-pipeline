const money = value => `$${Number(value).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
const number = value => Number(value).toLocaleString();
let dashboard = null;

async function refreshData() {
  const top = Math.max(1, Number(document.querySelector("#top-input").value || 20));
  const search = encodeURIComponent(document.querySelector("#search-input").value || "");
  const response = await fetch(`/api/dashboard?top=${top}&search=${search}`);
  if (!response.ok) return;
  dashboard = await response.json();
  document.querySelector("#orders").textContent = number(dashboard.metrics.orders);
  document.querySelector("#products").textContent = number(dashboard.metrics.products);
  document.querySelector("#revenue").textContent = money(dashboard.metrics.revenue);
  document.querySelector("#average-order").textContent = money(dashboard.metrics.average_order);
  renderCategories(dashboard.categories);
  renderMonthly(dashboard.monthly);
  renderCustomers(dashboard.top_customers, dashboard.metrics.revenue);
}

function renderCategories(rows) {
  const max = Math.max(...rows.map(row => row.revenue));
  document.querySelector("#category-chart").innerHTML = rows.map(row => `<div class="bar-row"><span>${row.category}</span><div class="bar-track"><div class="bar-fill" style="width:${row.revenue / max * 100}%"></div></div><span class="bar-value">${money(row.revenue)}</span></div>`).join("");
}

function renderMonthly(rows) {
  const max = Math.max(...rows.map(row => row.revenue));
  document.querySelector("#monthly-chart").innerHTML = rows.map(row => `<div class="line-column"><div class="line-bar" title="${row.month}: ${money(row.revenue)}" style="height:${row.revenue / max * 100}%"></div><span class="line-label">${row.month.slice(2)}</span></div>`).join("");
}

function renderCustomers(rows, total) {
  document.querySelector("#customers-table").innerHTML = rows.length ? rows.map((row, index) => `<tr><td><b>${index + 1}</b></td><td>${row.customer_id}</td><td>${money(row.lifetime_value)}</td><td>${number(row.order_count)}</td><td><span class="share-track"><span class="share-fill" style="width:${Math.min(100, row.lifetime_value / total * 100 * 8)}%"></span></span></td></tr>`).join("") : '<tr><td colspan="5" class="empty">No customers match this search.</td></tr>';
}

async function runBenchmark() {
  const status = document.querySelector("#benchmark-status");
  status.textContent = "Running both engines — this may take a moment...";
  const response = await fetch("/api/benchmark");
  if (!response.ok) { status.textContent = "Benchmark failed. Check the server logs."; return; }
  const data = await response.json();
  document.querySelector("#benchmark-table").innerHTML = data.results.map(row => `<tr><td><b>${row.Engine}</b></td><td>${row["Load (s)"].toFixed(2)}s</td><td>${row["Join + revenue (s)"].toFixed(2)}s</td><td>${row["Group + top 10 (s)"].toFixed(2)}s</td><td><b>${row["Total (s)"].toFixed(2)}s</b></td></tr>`).join("");
  status.textContent = "Benchmark complete.";
}

refreshData();
