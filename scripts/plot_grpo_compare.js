#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const defaultOutput = "/home/leejt/.codex/visualizations/2026/06/04/019e9271-a988-7422-a2c1-bcb868cdf53f/grpo-seed42-666-curves.html";
const outputPath = process.argv[2] || defaultOutput;
const inputPaths = process.argv.slice(3);

if (inputPaths.length < 2) {
  throw new Error("Pass at least two metrics JSONL files after the output path.");
}

function readRun(inputPath) {
  const rows = fs
    .readFileSync(inputPath, "utf8")
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const match = path.basename(inputPath).match(/metrics_seed(.+)\.jsonl$/);
  return {
    seed: match ? match[1] : path.basename(inputPath),
    data: rows,
  };
}

const runs = inputPaths.map(readRun);
const seedTitle = runs.map((run) => "seed " + run.seed).join(" + ");
const serializedRuns = JSON.stringify(runs);

const html = String.raw`<div id="grpo-seed42-666-curves">
  <style>
    #grpo-seed42-666-curves {
      --foreground: light-dark(#24313b, #edf2f3);
      --muted: light-dark(#66737d, #b4c0c4);
      --border: light-dark(#cbd4d8, #526066);
      --grid: light-dark(#e7edef, #344149);
      --popover: light-dark(#ffffff, #1f272b);
      --popover-foreground: light-dark(#24313b, #edf2f3);
      --viz-series-1: light-dark(#147d92, #56c2d4);
      --viz-series-2: light-dark(#d05a45, #f18b72);
      --viz-series-3: light-dark(#6b5ca5, #b2a5ed);
      --viz-series-4: light-dark(#c28a25, #e9bd57);
      --viz-series-5: light-dark(#39805a, #78c692);
      --viz-series-6: light-dark(#b04d78, #ea8eb5);
      color-scheme: light dark;
      color: var(--foreground);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      width: 100%;
      max-width: 1040px;
      margin: 0 auto;
      box-sizing: border-box;
    }
    #grpo-seed42-666-curves .heading {
      margin-bottom: 12px;
    }
    #grpo-seed42-666-curves h1 {
      font-size: 21px;
      line-height: 1.25;
      margin: 0;
      font-weight: 500;
      overflow-wrap: anywhere;
    }
    #grpo-seed42-666-curves .subtitle,
    #grpo-seed42-666-curves .line-key {
      color: var(--muted);
      font-size: 13px;
      margin: 4px 0 0;
    }
    #grpo-seed42-666-curves .legend {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 4px 14px;
      margin: 0 0 12px;
    }
    #grpo-seed42-666-curves .legend button {
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
    #grpo-seed42-666-curves .legend button[aria-pressed="false"] {
      opacity: 0.35;
    }
    #grpo-seed42-666-curves .swatch {
      background: currentColor;
      display: inline-block;
      height: 3px;
      width: 17px;
    }
    #grpo-seed42-666-curves .legend-note {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    #grpo-seed42-666-curves .plots {
      display: grid;
      gap: 22px 24px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    #grpo-seed42-666-curves .plot-panel {
      min-width: 0;
    }
    #grpo-seed42-666-curves .plot-title {
      font-size: 14px;
      font-weight: 500;
      margin: 0 0 2px;
    }
    #grpo-seed42-666-curves svg {
      display: block;
      height: auto;
      overflow: visible;
      width: 100%;
    }
    #grpo-seed42-666-curves .axis path,
    #grpo-seed42-666-curves .axis line {
      stroke: var(--border);
      shape-rendering: crispEdges;
    }
    #grpo-seed42-666-curves .axis text,
    #grpo-seed42-666-curves .axis-title {
      fill: var(--foreground);
      font-size: 12px;
    }
    #grpo-seed42-666-curves .grid-line {
      stroke: var(--grid);
      shape-rendering: crispEdges;
    }
    #grpo-seed42-666-curves .series-line {
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      stroke-width: 2;
    }
    #grpo-seed42-666-curves .series-point {
      stroke: var(--popover);
      stroke-width: 1.5;
    }
    #grpo-seed42-666-curves .hover-guide {
      stroke: var(--muted);
      stroke-dasharray: 3 3;
      stroke-width: 1;
    }
    #grpo-seed42-666-curves .tooltip {
      background: var(--popover);
      border: 1px solid var(--border);
      color: var(--popover-foreground);
      display: none;
      font-size: 12px;
      line-height: 1.45;
      max-width: 235px;
      padding: 7px 9px;
      pointer-events: none;
      position: absolute;
      z-index: 2;
    }
    #grpo-seed42-666-curves .tooltip-step {
      font-weight: 500;
      margin-bottom: 2px;
    }
    @media (max-width: 760px) {
      #grpo-seed42-666-curves .plots {
        grid-template-columns: minmax(0, 1fr);
      }
      #grpo-seed42-666-curves .legend-note {
        flex-basis: 100%;
      }
    }
    @media (max-width: 400px) {
      #grpo-seed42-666-curves .plot-panel svg {
        max-width: 320px;
      }
    }
  </style>
  <div class="heading">
    <h1>GRPO standard on-policy: ${seedTitle}</h1>
    <p class="subtitle">200 rollout steps · validation evaluated every 10 steps</p>
  </div>
  <div class="legend" aria-label="Runs"></div>
  <div class="plots"></div>
  <div class="tooltip" role="tooltip"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("grpo-seed42-666-curves");
  const runs = ${serializedRuns};
  const colors = ["var(--viz-series-1)", "var(--viz-series-2)", "var(--viz-series-3)", "var(--viz-series-4)", "var(--viz-series-5)", "var(--viz-series-6)"];
  const panels = [
    {
      id: "rewards",
      title: "Rewards",
      yLabel: "reward",
      metrics: [
        { field: "mean_reward", label: "train", points: false },
        { field: "val_reward", label: "validation", dash: "5 3", points: true }
      ]
    },
    {
      id: "format",
      title: "Format reward",
      yLabel: "reward",
      metrics: [
        { field: "mean_format_reward", label: "train", points: false },
        { field: "val_format_reward", label: "validation", dash: "5 3", points: true }
      ]
    },
    { id: "loss", title: "Policy-gradient loss", yLabel: "loss", metrics: [{ field: "loss", label: "train", points: false }] },
    { id: "grad", title: "Gradient norm", yLabel: "norm", metrics: [{ field: "grad_norm", label: "train", points: false }] },
    { id: "entropy", title: "Token entropy", yLabel: "entropy", metrics: [{ field: "token_entropy", label: "train", points: false }] },
    { id: "length", title: "Validation response length", yLabel: "tokens", metrics: [{ field: "val_avg_response_length", label: "validation", dash: "5 3", points: true }] }
  ];
  const enabledRuns = new Set(runs.map((run) => run.seed));
  const tooltip = d3.select(root).select(".tooltip");

  const legend = d3.select(root).select(".legend");
  legend.selectAll("button")
    .data(runs)
    .join("button")
    .attr("type", "button")
    .attr("aria-pressed", "true")
    .on("click", function(event, run) {
      if (enabledRuns.has(run.seed)) enabledRuns.delete(run.seed);
      else enabledRuns.add(run.seed);
      d3.select(this).attr("aria-pressed", enabledRuns.has(run.seed) ? "true" : "false");
      drawAll();
    })
    .each(function(run, index) {
      const button = d3.select(this);
      button.append("span").attr("class", "swatch").style("color", colors[index % colors.length]);
      button.append("span").text("seed " + run.seed);
    });
  legend.append("span").attr("class", "legend-note").text("solid = train · dashed with points = validation");

  const plotHosts = d3.select(root).select(".plots")
    .selectAll("section")
    .data(panels)
    .join("section")
    .attr("class", "plot-panel")
    .attr("data-panel", (panel) => panel.id);
  plotHosts.append("h2").attr("class", "plot-title").text((panel) => panel.title);
  plotHosts.append("svg").attr("role", "img").attr("aria-label", (panel) => panel.title);

  function pointsFor(run, field) {
    return run.data.filter((row) => Number.isFinite(Number(row[field])));
  }

  function interpolate(points, step, field) {
    if (points.length === 0) return null;
    const nearest = d3.bisector((row) => row.step).center(points, step);
    if (points.length === 1 || nearest === 0) return Number(points[0][field]);
    if (nearest === points.length - 1) return Number(points[points.length - 1][field]);
    const right = d3.bisector((row) => row.step).left(points, step);
    const leftRow = points[Math.max(0, right - 1)];
    const rightRow = points[Math.min(points.length - 1, right)];
    if (leftRow.step === rightRow.step) return Number(leftRow[field]);
    const ratio = (step - leftRow.step) / (rightRow.step - leftRow.step);
    return Number(leftRow[field]) + ratio * (Number(rightRow[field]) - Number(leftRow[field]));
  }

  function drawPanel(panel) {
    const host = root.querySelector('[data-panel="' + panel.id + '"]');
    const svg = d3.select(host).select("svg");
    svg.selectAll("*").remove();
    const measuredWidth = host.getBoundingClientRect().width || root.clientWidth || 320;
    const viewportWidth = document.documentElement.clientWidth || measuredWidth;
    const width = window.matchMedia("(max-width: 400px)").matches
      ? Math.min(320, Math.max(280, viewportWidth))
      : Math.max(280, Math.min(measuredWidth, viewportWidth));
    const height = 248;
    const margin = { top: 8, right: 12, bottom: 38, left: 54 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    svg.attr("viewBox", "0 0 " + width + " " + height);

    const x = d3.scaleLinear()
      .domain(d3.extent(runs.flatMap((run) => run.data), (row) => row.step))
      .range([margin.left, width - margin.right]);
    const values = runs.flatMap((run) => panel.metrics.flatMap((metric) =>
      pointsFor(run, metric.field).map((row) => Number(row[metric.field]))
    ));
    const extent = d3.extent(values);
    const span = extent[1] - extent[0] || Math.max(1, Math.abs(extent[0] || 1));
    const y = d3.scaleLinear()
      .domain([extent[0] - span * 0.08, extent[1] + span * 0.08])
      .nice()
      .range([height - margin.bottom, margin.top]);

    const chart = svg.append("g");
    chart.append("rect")
      .attr("data-chart-frame", "true")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", "none")
      .attr("stroke", "var(--border)");
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
    runs.filter((run) => enabledRuns.has(run.seed)).forEach((run, runIndex) => {
      const color = colors[runIndex % colors.length];
      panel.metrics.forEach((metric) => {
        const points = pointsFor(run, metric.field)
          .map((row) => ({ step: row.step, value: Number(row[metric.field]) }));
        if (points.length === 0) return;
        chart.append("path")
          .datum(points)
          .attr("class", "series-line")
          .attr("data-series", run.seed + "-" + metric.field)
          .attr("stroke", color)
          .attr("stroke-dasharray", metric.dash || null)
          .attr("d", line);
        if (metric.points && points.length <= 25) {
          chart.selectAll(".point-" + runIndex + "-" + metric.field.replaceAll("_", "-"))
            .data(points)
            .join("circle")
            .attr("class", "series-point point-" + runIndex + "-" + metric.field.replaceAll("_", "-"))
            .attr("data-series", run.seed + "-" + metric.field)
            .attr("cx", (row) => x(row.step))
            .attr("cy", (row) => y(row.value))
            .attr("r", 3)
            .attr("fill", color);
        }
      });
    });

    const guide = chart.append("line")
      .attr("class", "hover-guide")
      .attr("data-chart-hover-guide", "true")
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .style("display", "none");
    const markerLayer = chart.append("g").attr("class", "hover-markers");
    chart.append("rect")
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
        const rows = runs.filter((run) => enabledRuns.has(run.seed)).flatMap((run, runIndex) =>
          panel.metrics.map((metric) => {
            const value = interpolate(pointsFor(run, metric.field), step, metric.field);
            return value == null ? null : {
              color: colors[runIndex % colors.length],
              label: "seed " + run.seed + " · " + metric.label,
              value,
            };
          })
        ).filter(Boolean);
        guide.attr("x1", x(step)).attr("x2", x(step)).style("display", null);
        markerLayer.selectAll("circle")
          .data(rows)
          .join("circle")
          .attr("data-chart-hover-marker", "true")
          .attr("cx", x(step))
          .attr("cy", (row) => y(row.value))
          .attr("r", 4)
          .attr("fill", (row) => row.color)
          .attr("stroke", "var(--popover)")
          .attr("stroke-width", 1.5);
        const box = root.getBoundingClientRect();
        tooltip
          .style("display", "block")
          .style("left", Math.min(event.clientX - box.left + 12, box.width - 247) + "px")
          .style("top", Math.max(0, event.clientY - box.top - 12) + "px")
          .html("<div class=\"tooltip-step\">step " + Math.round(step) + "</div>" + rows
            .map((row) => "<div><span style=\"color:" + row.color + "\">" + row.label + "</span>: " + d3.format(".3f")(row.value) + "</div>")
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
