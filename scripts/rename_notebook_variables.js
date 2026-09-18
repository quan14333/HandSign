const fs = require("fs");

const notebookPath = process.argv[2];
if (!notebookPath) {
  throw new Error("Usage: node rename_notebook_variables.js <notebook.ipynb>");
}

const backupPath = notebookPath.replace(/\.ipynb$/i, ".before-variable-rename.ipynb");
if (!fs.existsSync(backupPath)) {
  throw new Error(`Backup not found: ${backupPath}`);
}

// Start from the untouched backup, then rename identifiers only.
const notebook = JSON.parse(fs.readFileSync(backupPath, "utf8"));

function renameCell(cellIndex, renames) {
  const cell = notebook.cells[cellIndex];
  if (!cell || cell.cell_type !== "code") {
    throw new Error(`Expected code cell at index ${cellIndex}`);
  }

  let source = cell.source.join("");
  for (const [oldName, newName] of Object.entries(renames)) {
    source = source.replace(new RegExp(`\\b${oldName}\\b`, "g"), newName);
  }
  cell.source = source.match(/[^\n]*\n|[^\n]+$/g) || [];
}

const linearRenames = {
  history_w: "history_w_linear",
  history_b: "history_b_linear",
  y_pred: "y_pred_linear",
  weight: "weight_linear",
  bias: "bias_linear",
  dw: "dw_linear",
  db: "db_linear",
  x: "x_linear",
  y: "y_linear"
};

for (const cellIndex of [2, 3, 4]) {
  renameCell(cellIndex, linearRenames);
}

const logisticRenames = {
  his_w: "his_w_logistic",
  his_b: "his_b_logistic",
  y_pred: "y_pred_logistic",
  x0: "x0_logistic",
  x1: "x1_logistic",
  X: "X_logistic",
  y: "y_logistic",
  w: "w_logistic",
  b: "b_logistic",
  dw: "dw_logistic",
  db: "db_logistic"
};

for (const cellIndex of [10, 11, 12, 13, 14]) {
  renameCell(cellIndex, logisticRenames);
}

const softmaxRenames = {
  h_w: "h_w_softmax",
  h_b: "h_b_softmax",
  y_pred: "y_pred_softmax",
  x0: "x0_softmax",
  x1: "x1_softmax",
  x2: "x2_softmax",
  x3: "x3_softmax",
  x4: "x4_softmax",
  X: "X_softmax",
  Y: "Y_softmax",
  y: "y_softmax",
  W: "W_softmax",
  w: "w_softmax",
  b: "b_softmax",
  dW: "dW_softmax",
  db: "db_softmax"
};

for (const cellIndex of [18, 19, 20, 21, 22, 23]) {
  renameCell(cellIndex, softmaxRenames);
}

fs.writeFileSync(notebookPath, `${JSON.stringify(notebook, null, 1)}\n`, "utf8");
console.log(`Restored original structure and applied suffix-only renames: ${notebookPath}`);
