import React, { useEffect, useMemo, useState } from "react";
import { uiText } from "../i18n/languages";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingFn,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react";
import { apiClient, type AITableSpec } from "../api/client";
import { formatTableCell, numericValueOf } from "../utils/formatters";
import { tableToPayload } from "../utils/clipboard";
import { CopyButton } from "./CopyButton";

type TableRow = Record<string, unknown>;

/**
 * Cells arrive as strings the backend already formatted ("1,234.00"), so the
 * default lexicographic comparator would order them wrongly -- "9" after "10".
 * This compares the underlying numbers when both cells are numeric and falls
 * back to a natural string compare otherwise. It only changes row *order*; the
 * displayed text is always the backend's.
 */
const numericAwareSort: SortingFn<TableRow> = (rowA, rowB, columnId) => {
  const rawA = rowA.getValue(columnId);
  const rawB = rowB.getValue(columnId);
  const numA = numericValueOf(rawA);
  const numB = numericValueOf(rawB);

  if (numA !== null && numB !== null) return numA - numB;
  if (numA !== null) return -1;
  if (numB !== null) return 1;
  return String(rawA ?? "").localeCompare(String(rawB ?? ""), undefined, {
    numeric: true,
    sensitivity: "base",
  });
};

interface ResultTableProps {
  table: AITableSpec;
  /** Backend trace id of the answer -- lets pages beyond the delivered rows be
   * fetched server-side (POST /api/table-page) instead of loading everything. */
  traceId?: string;
  tableIndex?: number;
  /** The answer's language: headers (table.column_labels) and controls follow it. */
  lang?: string;
}

const PAGE_SIZES = [25, 50, 100];

export const ResultTable: React.FC<ResultTableProps> = ({ table, traceId, tableIndex = 0, lang }) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [remoteRows, setRemoteRows] = useState<TableRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);

  const columns = useMemo<ColumnDef<TableRow>[]>(
    () =>
      table.columns.map((name) => ({
        id: name,
        // accessorFn, not accessorKey: these ids are human header labels that
        // can contain dots and spaces, which accessorKey would parse as a deep
        // property path and fail to resolve.
        accessorFn: (row: TableRow) => row[name],
        // Shown in the answer's language when translated; the row key stays the column.
        header: (table.column_labels?.[name] || name.replace(/_/g, " ")).toUpperCase(),
        cell: (info) => formatTableCell(info.getValue()),
        sortingFn: numericAwareSort,
      })),
    [table.columns, table.column_labels]
  );

  // A grand-total row (first cell "Total", e.g. the month-wise installation
  // table) is pulled out of the sortable body and pinned as the footer, so
  // sorting the months never buries it mid-table.
  const firstColumn = table.columns[0];
  const isTotal = (row: TableRow) =>
    firstColumn !== undefined && String(row[firstColumn] ?? "").trim().toLowerCase() === "total";
  const { delivered, totalRow } = useMemo(() => {
    const all = table.rows ?? [];
    const total = all.length > 1 ? all.find(isTotal) : undefined;
    return { delivered: total ? all.filter((row) => row !== total) : all, totalRow: total };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table.rows, firstColumn]);

  // Pagination: rows already delivered are paged locally (instant); pages beyond them
  // come from the server, which re-runs the answer's validated SQL with OFFSET / LIMIT.
  const totalCount = Math.max(table.total_rows - (totalRow ? 1 : 0), delivered.length);
  // Page 1 is exactly what the answer showed (up to a page); later pages continue right
  // after it -- a 'top 5' answer shows its 5, then 'Next' brings rows 6 onward.
  const firstPageRows = Math.min(delivered.length || pageSize, pageSize);
  const totalPages = Math.max(1, 1 + Math.ceil(Math.max(0, totalCount - firstPageRows) / pageSize));
  const pageStart = page === 1 ? 0 : firstPageRows + (page - 2) * pageSize;
  const pageRows = page === 1 ? firstPageRows : pageSize;
  const needsServer = pageStart + pageRows > delivered.length && delivered.length < totalCount && !!traceId;

  useEffect(() => {
    setPage(1);
  }, [pageSize, table.rows]);

  useEffect(() => {
    if (!needsServer || !traceId) {
      setRemoteRows(null);
      setPageError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setPageError(null);
    apiClient
      .tablePage(traceId, page, pageSize, tableIndex, pageStart)
      .then((res) => {
        if (cancelled) return;
        // Same SQL -> same column order: map positionally onto this table's header labels.
        const mapped = res.rows.map((r) => Object.fromEntries(table.columns.map((name, i) => [name, r[i]])));
        setRemoteRows(mapped.filter((row) => !isTotal(row)));
      })
      .catch((err: unknown) => {
        if (!cancelled) setPageError(err instanceof Error ? err.message : "This page could not be loaded.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsServer, traceId, page, pageSize, tableIndex, pageStart]);

  const data = useMemo(
    () => (needsServer ? remoteRows ?? [] : delivered.slice(pageStart, pageStart + pageRows)),
    [needsServer, remoteRows, delivered, pageStart, pageRows]
  );
  const showPager = totalCount > PAGE_SIZES[0];

  // A column is right-aligned when every non-empty cell in it is a number.
  const numericColumns = useMemo(() => {
    const out = new Set<string>();
    for (const name of table.columns) {
      const values = data.map((row) => row[name]).filter((v) => v !== null && v !== undefined && v !== "");
      if (values.length && values.every((v) => numericValueOf(v) !== null)) out.add(name);
    }
    return out;
  }, [data, table.columns]);

  const instance = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const rows = instance.getRowModel().rows;

  return (
    <div className="table-card">
      <div className="table-card-toolbar">
        <div>
          <h3 className="table-card-title">{table.display_title || table.title}</h3>
          <p className="table-card-subtitle">
            {showPager
              ? uiText(lang, "showingRows", {
                  from: (pageStart + 1).toLocaleString(),
                  to: Math.min(pageStart + (data.length || pageRows), totalCount).toLocaleString(),
                  total: totalCount.toLocaleString(),
                })
              : uiText(lang, "totalRows", { total: totalCount.toLocaleString() })}
          </p>
        </div>
        <CopyButton
          getPayload={() => tableToPayload(table)}
          label={uiText(lang, "copyTable")}
          title={uiText(lang, "copyTableTitle")}
          copiedLabel={uiText(lang, "copied")}
        />
      </div>
      <div className="table-container">
        <table className="data-table">
          <thead>
            {instance.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => {
                  const sorted = header.column.getIsSorted();
                  return (
                    <th key={header.id} className={numericColumns.has(header.column.id) ? "col-numeric" : undefined}>
                      <button
                        type="button"
                        className="th-sort-btn"
                        onClick={header.column.getToggleSortingHandler()}
                        title={uiText(lang, "sortColumn")}
                      >
                        <span>{flexRender(header.column.columnDef.header, header.getContext())}</span>
                        {sorted === "asc" ? (
                          <ArrowUp size={12} />
                        ) : sorted === "desc" ? (
                          <ArrowDown size={12} />
                        ) : (
                          <ArrowUpDown size={12} className="th-sort-idle" />
                        )}
                      </button>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className={numericColumns.has(cell.column.id) ? "col-numeric" : undefined}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          {totalRow && (
            <tfoot>
              <tr className="data-table-total">
                {table.columns.map((name) => (
                  <td key={name} className={numericColumns.has(name) ? "col-numeric" : undefined}>
                    {formatTableCell(totalRow[name])}
                  </td>
                ))}
              </tr>
            </tfoot>
          )}
        </table>
        {loading ? <div className="table-page-status">{uiText(lang, "loadingPage")}</div> : null}
        {pageError ? <div className="table-page-status table-page-error">{pageError}</div> : null}
        {!loading && !pageError && data.length === 0 ? <div className="table-page-status">{uiText(lang, "noRowsPage")}</div> : null}
      </div>
      {showPager ? (
        <div className="table-pager">
          <label className="table-pager-size">
            {uiText(lang, "rowsPerPage")}
            <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
              {PAGE_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <div className="table-pager-nav">
            <button type="button" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1 || loading}
              aria-label={uiText(lang, "previous")}>
              <ChevronLeft size={14} /> {uiText(lang, "previous")}
            </button>
            <span className="table-pager-page">
              {uiText(lang, "pageOf", { page, pages: totalPages })}
            </span>
            <button type="button" onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading || (!traceId && pageStart + pageRows >= delivered.length)}
              aria-label={uiText(lang, "next")}>
              {uiText(lang, "next")} <ChevronRight size={14} />
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
};
