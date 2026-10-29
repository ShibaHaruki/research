// 参照テンプレートのシート・書式を保ち、Python側で集計した数値を差し替える。
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [payload, modules] = process.argv.slice(2);
const require = createRequire(path.join(modules, '_lsm_loader.cjs'));
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
let inputText;
if (payload === '--stdin') {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  inputText = Buffer.concat(chunks).toString('utf8');
} else {
  inputText = await fs.readFile(payload, 'utf8');
}
const books = JSON.parse(inputText);
for (const book of books) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(book.template));
  for (const data of book.sheets) {
    const sheet = workbook.worksheets.getItem(data.name);
    for (const block of data.blocks) {
      const rows = block.rows.length;
      const cols = block.rows[0].length;
      const destination = sheet.getRangeByIndexes(block.row, 0, rows, cols);
      // 複数条件を下に追加するときも、最初の表と同じ書式にする。
      if (block.row > 0) destination.copyFrom(sheet.getRangeByIndexes(0, 0, rows, cols), 'all');
      destination.values = block.rows;
    }
  }
  // Excelの文字列定数の長さ制限に合わせ、再開確認用の情報を分割保存する。
  if (book.metadata) {
    const encoded = Buffer.from(JSON.stringify(book.metadata), 'utf8').toString('base64');
    for (let offset = 0; offset < encoded.length; offset += 200) {
      const name = `_LSM_EVAL_${String(offset / 200).padStart(4, '0')}`;
      workbook.names.add(name, `="${encoded.slice(offset, offset + 200)}"`);
    }
  }
  workbook.recalculate();
  const output = await SpreadsheetFile.exportXlsx(workbook);
  // 保存失敗時に既存の評価ファイルを壊さないよう、一時ファイルから置き換える。
  const temporary = `${book.path}.${process.pid}.tmp.xlsx`;
  try {
    await output.save(temporary);
    await fs.rename(temporary, book.path);
  } finally {
    await fs.rm(temporary, { force: true });
    // artifact-toolの自動診断ファイルをevalへ残さない。
    await fs.rm(`${temporary}.inspect.ndjson`, { force: true });
  }
  if (process.env.LSM_EXCEL_PREVIEW_DIR) {
    const dir = process.env.LSM_EXCEL_PREVIEW_DIR;
    await fs.mkdir(dir, { recursive: true });
    for (const [name, range] of [['conf_8cls', 'A1:I9'], ['conf_3cls', 'A1:C4'], ['accuracy', 'A1:I2']]) {
      const preview = await workbook.render({ sheetName: name, range, scale: 1.5 });
      await fs.writeFile(path.join(dir, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
    }
    console.log((await workbook.inspect({kind: 'table', range: 'accuracy!A1:I2',
      include: 'values,formulas', tableMaxRows: 2, tableMaxCols: 9})).ndjson);
  }
}
