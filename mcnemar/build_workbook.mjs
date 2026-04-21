import fs from "node:fs/promises";
import path from "node:path";

const artifactToolModule = process.env.ARTIFACT_TOOL_PATH || "@oai/artifact-tool";
const { Workbook, SpreadsheetFile } = await import(artifactToolModule);

const [, , payloadPath, outputPath] = process.argv;

if (!payloadPath || !outputPath) {
  throw new Error("Usage: node build_workbook.mjs <payload.json> <output.xlsx>");
}

function columnLabel(columnNumber) {
  let n = columnNumber;
  let label = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    n = Math.floor((n - 1) / 26);
  }
  return label;
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();

for (const sheetDef of payload.sheets) {
  const worksheet = workbook.worksheets.add(sheetDef.name);
  const rows = sheetDef.rows ?? [];
  if (rows.length > 0) {
    const width = Math.max(...rows.map((row) => row.length), 1);
    const normalizedRows = rows.map((row) => [...row, ...Array(width - row.length).fill(null)]);
    worksheet.getRange(`A1:${columnLabel(width)}${normalizedRows.length}`).values = normalizedRows;
  }
  for (const formatDef of sheetDef.formats ?? []) {
    worksheet.getRange(formatDef.range).setNumberFormat(formatDef.numberFormat);
  }
}

const inspection = await workbook.inspect({
  kind: "table",
  range: "Article Exact Match!A1:T13",
  include: "values",
  tableMaxRows: 13,
  tableMaxCols: 20,
});
console.log(inspection.ndjson);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
