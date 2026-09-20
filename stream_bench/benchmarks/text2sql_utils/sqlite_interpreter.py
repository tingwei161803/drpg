import resource
import sqlite3
import sys
from contextlib import closing

from func_timeout import FunctionTimedOut, func_timeout


def limit_memory(max_mem_gb):
    """Cap the memory the current process may allocate (Linux only)."""
    max_bytes = max_mem_gb * 1024 * 1024 * 1024  # GB → Bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

def execute_sql(predicted_sql, ground_truth, db_path, show_num_rows=10):
    limit_memory(10)  # cap at 10 GB of RAM

    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()

        # Predicted SQL query
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        if predicted_res:
            pred_md_table = "| " + " | ".join([f"Column{i+1}" for i in range(len(predicted_res[0]))]) + " |\n"
            pred_md_table += "| " + " | ".join(["---"] * len(predicted_res[0])) + " |\n"
            for row in predicted_res[:show_num_rows]:
                pred_md_table += "| " + " | ".join(map(str, row)) + " |\n"
        else:
            pred_md_table = "No result returned.\n"

        # Ground-truth SQL query
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        if ground_truth_res:
            gt_md_table = "| " + " | ".join([f"Column{i+1}" for i in range(len(ground_truth_res[0]))]) + " |\n"
            gt_md_table += "| " + " | ".join(["---"] * len(ground_truth_res[0])) + " |\n"
            for row in ground_truth_res[:show_num_rows]:
                gt_md_table += "| " + " | ".join(map(str, row)) + " |\n"
        else:
            gt_md_table = "No result returned.\n"

        res = int(set(predicted_res) == set(ground_truth_res))
        return res, pred_md_table, gt_md_table

def execute_model(predicted_sql, ground_truth, db_path, meta_time_out=30):
    pred_md_table, gt_md_table = '', ''
    try:
        res, pred_md_table, gt_md_table = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_path)
        )
        result = pred_md_table
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = 'Database execution timeout'
        res = 0
    except MemoryError:
        result = 'Memory limit (20GB) exceeded'
        res = 0
    except Exception as e:
        result = str(e)
        res = 0
    return res, (result, gt_md_table)
