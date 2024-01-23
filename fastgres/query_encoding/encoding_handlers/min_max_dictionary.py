
from fastgres.baseline.log_utils import get_logger
from fastgres.baseline.database_connection import DatabaseConnection


def build_db_min_max(db_connection: DatabaseConnection) -> dict:
    unhandled = set()
    conn, cursor = db_connection.establish_connection()
    cursor.execute("SELECT table_name "
                   "FROM information_schema.tables "
                   "WHERE table_schema = 'public'")
    mm_dict = dict()
    for table in cursor.fetchall():
        t = table[0]
        cursor.execute(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name = '{}';".format(t))
        col_dict = dict()
        for column, d_type in cursor.fetchall():
            if d_type in ['integer', 'timestamp without time zone', 'date', 'numeric']:
                cursor.execute("SELECT min({}), max({}) "
                               "FROM {};".format(column, column, t))
                mm_val = list(cursor.fetchall()[0])
                col_dict[column] = mm_val
            else:
                unhandled.add(d_type)
        mm_dict[t] = col_dict
    cursor.close()
    conn.close()
    logger = get_logger()
    logger.info(f'Unhandled d_types (mm_dict): {unhandled}')
    return mm_dict
