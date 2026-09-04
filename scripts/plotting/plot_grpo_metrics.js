#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const inputPath = process.argv[2] || "experiments/on_policy/standard/metrics_seed42.jsonl";
const outputPath = process.argv[3] || "experiments/figures/on_policy/metrics_seed42.html";

const rows = fs
  .readFileSync(inputPath, "utf8")
  .trim()
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const data = JSON.stringify(rows);

const html = String.raw`<div id="grpo-seed42-curves">
  <style>
    #grpo-seed42-curves {
      --foreground: #24313b;
      --muted: #66737d;
      --border: #cbd4d8;
      --grid: #e7edef;
      --popover: #ffffff;
      --popover-foreground: #24313b;
      --viz-series-1: #147d92;
      --viz-series-2: #d05a45;
      --viz-series-3: #6b5ca5;
      --viz-series-4: #c28a25;
      --viz-series-5: #39805a;
      --viz-series-6: #b04d78;
      color: var(--foreground);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      max-width: 1040px;
      margin: 0 auto;
    }
    #grpo-seed42-curves .heading {
      margin-bottom: 12px;
    }
    #grpo-seed42-curves h1 {
      font-size: 21px;
      line-height: 1.25;
      margin: 0;
      font-weight: 700;
    }
    #grpo-seed42-curves .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin: 4px 0 0;
    }
    #grpo-seed42-curves .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 4px 14px;
      margin: 0 0 12px;
    }
    #grpo-seed42-curves .legend button {
      align-items: center;
      background: transparent;
      border: 0;
      color: var(--foreground);
      cursor: pointer;
      display: inline-flex;
      font: inherit;
      font-size: 12px;
      gap: 5px;
      opacity: 1;
      padding: 2px 0;
    }
    #grpo-seed42-curves .legend button[aria-pressed="false"] {
      opacity: 0.35;
    }
    #grpo-seed42-curves .swatch {
      background: currentColor;
      display: inline-block;
      height: 3px;
      width: 17px;
    }
    #grpo-seed42-curves .plots {
      display: grid;
      gap: 22px 24px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    #grpo-seed42-curves .plot-panel {
      min-width: 0;
    }
    #grpo-seed42-curves .plot-title {
      font-size: 14px;
      font-weight: 650;
      margin: 0 0 2px;
    }
    #grpo-seed42-curves svg {
      display: block;
      height: auto;
      overflow: visible;
      width: 100%;
    }
    #grpo-seed42-curves .axis path,
    #grpo-seed42-curves .axis line {
      stroke: var(--border);
      shape-rendering: crispEdges;
    }
    #grpo-seed42-curves .axis text,
    #grpo-seed42-curves .axis-title {
      fill: var(--foreground);
      font-size: 12px;
    }
    #grpo-seed42-curves .grid-line {
      stroke: var(--grid);
      shape-rendering: crispEdges;
    }
    #grpo-seed42-curves .series-line {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 2;
    }
    #grpo-seed42-curves .series-point {
      stroke: var(--popover);
      stroke-width: 1.5;
    }
    #grpo-seed42-curves .hover-guide {
      stroke: var(--muted);
      stroke-dasharray: 3 3;
      stroke-width: 1;
    }
    #grpo-seed42-curves .tooltip {
      background: var(--popover);
      border: 1px solid var(--border);
      color: var(--popover-foreground);
      display: none;
      font-size: 12px;
      line-height: 1.45;
      max-width: 220px;
      padding: 7px 9px;
      pointer-events: none;
      position: absolute;
      z-index: 2;
    }
    #grpo-seed42-curves .tooltip .tooltip-step {
      font-weight: 700;
      margin-bottom: 2px;
    }
    @media (max-width: 760px) {
      #grpo-seed42-curves .plots {
        grid-template-columns: minmax(0, 1fr);
      }
    }
  </style>
  <div class="heading">
    <h1>GRPO standard on-policy: seed 42</h1>
    <p class="subtitle">200 rollout steps · validation evaluated every 10 steps</p>
  </div>
  <div class="legend" aria-label="Series"></div>
  <div class="plots"></div>
  <div class="tooltip" role="tooltip"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("grpo-seed42-curves");
  const data = ${data};
  const series = [
    { key: "mean_reward", label: "train reward", color: "var(--viz-series-1)" },
    { key: "val_reward", label: "val reward", color: "var(--viz-series-2)" },
    { key: "mean_format_reward", label: "train format", color: "var(--viz-series-3)" },
    { key: "val_format_reward", label: "val format", color: "var(--viz-series-4)" },
    { key: "loss", label: "loss", color: "var(--viz-series-5)" },
    { key: "grad_norm", label: "gradient norm", color: "var(--viz-series-6)" },
    { key: "token_entropy", label: "token entropy", color: "var(--viz-series-1)" },
    { key: "val_avg_response_length", label: "val response length", color: "var(--viz-series-4)" }
  ];
  const panels = [
    { id: "rewards", title: "Rewards", yLabel: "reward", keys: ["mean_reward", "val_reward"] },
    { id: "format", title: "Format reward", yLabel: "reward", keys: ["mean_format_reward", "val_format_reward"] },
    { id: "loss", title: "Policy-gradient loss", yLabel: "loss", keys: ["loss"] },
    { id: "grad", title: "Gradient norm", yLabel: "norm", keys: ["grad_norm"] },
    { id: "entropy", title: "Token entropy", yLabel: "entropy", keys: ["token_entropy"] },
    { id: "length", title: "Validation response length", yLabel: "tokens", keys: ["val_avg_response_length"] }
  ];
  const enabled = new Set(series.map((item) => item.key));
  const tooltip = d3.select(root).select(".tooltip");

  const legend = d3.select(root).select(".legend");
  legend.selectAll("button")
    .data(series)
    .join("button")
    .attr("type", "button")
    .attr("aria-pressed", "true")
    .on("click", function(event, item) {
      if (enabled.has(item.key)) enabled.delete(item.key);
      else enabled.add(item.key);
      d3.select(this).attr("aria-pressed", enabled.has(item.key) ? "true" : "false");
      drawAll();
    })
    .each(function(item) {
      const button = d3.select(this);
      button.append("span").attr("class", "swatch").style("color", item.color);
      button.append("span").text(item.label);
    });

  const plotHosts = d3.select(root).select(".plots")
    .selectAll("section")
    .data(panels)
    .join("section")
    .attr("class", "plot-panel")
    .attr("data-panel", (panel) => panel.id);
  plotHosts.append("h2").attr("class", "plot-title").text((panel) => panel.title);
  plotHosts.append("svg").attr("role", "img").attr("aria-label", (panel) => panel.title);

  function numberValues(keys) {
    return keys.flatMap((key) => data
      .map((row) => Number(row[key]))
      .filter((value) => Number.isFinite(value)));
  }

  function interpolate(points, step, key) {
    if (points.length === 0) return null;
    if (points.length === 1) return points[0][key];
    const right = d3.bisector((row) => row.step).left(points, step);
    if (right <= 0) return points[0][key];
    if (right >= points.length) return points[points.length - 1][key];
    const leftRow = points[right - 1];
    const rightRow = points[right];
    const ratio = (step - leftRow.step) / (rightRow.step - leftRow.step);
    return leftRow[key] + ratio * (rightRow[key] - leftRow[key]);
  }

  function drawPanel(panel) {
    const host = root.querySelector('[data-panel="' + panel.id + '"]');
    const svg = d3.select(host).select("svg");
    svg.selectAll("*").remove();
    const width = Math.max(280, host.clientWidth || 420);
    const height = 248;
    const margin = { top: 8, right: 12, bottom: 38, left: 54 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    svg.attr("viewBox", "0 0 " + width + " " + height);

    const x = d3.scaleLinear()
      .domain(d3.extent(data, (row) => row.step))
      .range([margin.left, width - margin.right]);
    const values = numberValues(panel.keys);
    const extent = d3.extent(values);
    const span = extent[1] - extent[0] || Math.max(1, Math.abs(extent[0] || 1));
    const y = d3.scaleLinear()
      .domain([extent[0] - span * 0.08, extent[1] + span * 0.08])
      .nice()
      .range([height - margin.bottom, margin.top]);

    const chart = svg.append("g");
    chart.append("g")
      .attr("class", "grid")
      .attr("transform", "translate(" + margin.left + ",0)")
      .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(""))
      .selectAll("line")
      .attr("class", "grid-line");
    chart.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(0," + (height - margin.bottom) + ")")
      .call(d3.axisBottom(x).ticks(width < 380 ? 4 : 6).tickFormat(d3.format("d")));
    chart.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(" + margin.left + ",0)")
      .call(d3.axisLeft(y).ticks(5));
    chart.append("text")
      .attr("class", "axis-title")
      .attr("data-axis", "x")
      .attr("x", margin.left + innerWidth / 2)
      .attr("y", height - 4)
      .attr("text-anchor", "middle")
      .text("rollout step");
    chart.append("text")
      .attr("class", "axis-title")
      .attr("data-axis", "y")
      .attr("transform", "translate(13," + (margin.top + innerHeight / 2) + ") rotate(-90)")
      .attr("text-anchor", "middle")
      .text(panel.yLabel);

    const line = d3.line()
      .defined((row) => Number.isFinite(Number(row.value)))
      .x((row) => x(row.step))
      .y((row) => y(row.value));
    panel.keys.filter((key) => enabled.has(key)).forEach((key) => {
      const item = series.find((entry) => entry.key === key);
      const points = data
        .filter((row) => Number.isFinite(Number(row[key])))
        .map((row) => ({ step: row.step, value: Number(row[key]) }));
      chart.append("path")
        .datum(points)
        .attr("class", "series-line")
        .attr("data-series", key)
        .attr("stroke", item.color)
        .attr("d", line);
      if (points.length <= 25) {
        chart.selectAll(".point-" + key.replaceAll("_", "-"))
          .data(points)
          .join("circle")
          .attr("class", "series-point point-" + key.replaceAll("_", "-"))
          .attr("data-series", key)
          .attr("cx", (row) => x(row.step))
          .attr("cy", (row) => y(row.value))
          .attr("r", 3)
          .attr("fill", item.color);
      }
    });

    const guide = chart.append("line")
      .attr("class", "hover-guide")
      .attr("data-chart-hover-guide", "true")
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .style("display", "none");
    const markerLayer = chart.append("g").attr("class", "hover-markers");
    const overlay = chart.append("rect")
      .attr("data-chart-hit", "true")
      .attr("data-chart-hover-overlay", "cross-series")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", "transparent")
      .on("pointermove", function(event) {
        const pointer = d3.pointer(event, this);
        const step = x.invert(pointer[0]);
        const rows = panel.keys
          .filter((key) => enabled.has(key))
          .map((key) => {
            const item = series.find((entry) => entry.key === key);
            const points = data.filter((row) => Number.isFinite(Number(row[key])));
            const value = interpolate(points, step, key);
            return value == null ? null : { item, value };
          })
          .filter(Boolean);
        guide.attr("x1", x(step)).attr("x2", x(step)).style("display", null);
        markerLayer.selectAll("circle")
          .data(rows)
          .join("circle")
          .attr("data-chart-hover-marker", "true")
          .attr("cx", x(step))
          .attr("cy", (row) => y(row.value))
          .attr("r", 4)
          .attr("fill", (row) => row.item.color)
          .attr("stroke", "var(--popover)")
          .attr("stroke-width", 1.5);
        const box = root.getBoundingClientRect();
        const eventBox = this.ownerSVGElement.getBoundingClientRect();
        tooltip
          .style("display", "block")
          .style("left", Math.min(event.clientX - box.left + 12, box.width - 232) + "px")
          .style("top", Math.max(0, event.clientY - box.top - 12) + "px")
          .html("<div class=\"tooltip-step\">step " + Math.round(step) + "</div>" + rows
            .map((row) => "<div><span style=\"color:" + row.item.color + "\">" + row.item.label + "</span>: " + d3.format(".3f")(row.value) + "</div>")
            .join(""));
      })
      .on("pointerleave", function() {
        guide.style("display", "none");
        markerLayer.selectAll("circle").remove();
        tooltip.style("display", "none");
      });
  }

  function drawAll() {
    panels.forEach(drawPanel);
  }

  drawAll();
  new ResizeObserver(drawAll).observe(root.querySelector(".plots"));
})();
</script>`;

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, html, "utf8");
console.log(outputPath);
