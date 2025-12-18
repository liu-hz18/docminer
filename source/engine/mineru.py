import os
from loguru import logger
from pathlib import Path
from mineru.cli.common import do_parse, read_fn


def parse_doc(input_path_list: list[Path], output_dir: Path, config: dict):
    old_device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = config["device"]
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ['ASCEND_RT_VISIBLE_DEVICES']}")
    lang = config["lang"]
    backend = config["backend"]
    parse_method = config["parse_method"]
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in input_path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
            server_url=None,
            start_page_id=0,
            end_page_id=None,
        )
    except Exception as e:
        logger.exception(e)
    finally:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_device
