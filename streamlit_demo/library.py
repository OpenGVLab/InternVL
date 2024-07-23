# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/hreikin/streamlit-uploads-library/blob/main/streamlit_uploads_library/library.py
# --------------------------------------------------------

import logging
from math import ceil

import streamlit as st

logger = logging.getLogger(__name__)


class Library():
    """Create a simple library out of streamlit widgets.

    Using the library is simple, import `streamlit_uploads_library` and then instantiate the class with the
    required `directory` variable. Other options can be configured by passing in different variables
    when instantiating the class.

    Example Usage:
        python
        import streamlit as st
        from library import Library

        st.set_page_config(page_title="Streamlit Uploads Library", layout="wide")
        default_library = Library(images=pil_images)
    """

    def __init__(self, images, image_alignment='end', number_of_columns=5):
        self.images = images
        self.image_alignment = image_alignment
        self.number_of_columns = number_of_columns
        self.root_container = self.create(images=self.images,
                                          image_alignment=self.image_alignment,
                                          number_of_columns=self.number_of_columns)

    def create(_self, images, image_alignment, number_of_columns):
        """Creates a simple library or gallery with columns.

        Creates a library or gallery using columns out of streamlit widgets.
        """
        root_container = st.container()
        with root_container:
            # To be able to display the images, details and buttons all in one row and aligned
            # correctly so that images of different sizes don't affect the alignment of the details
            # and buttons we need do some minor maths and keep track of multiple index values.
            # First we instantiate some defaults.
            col_idx = 0
            filename_idx = 0
            max_idx = number_of_columns - 1
            # Get the file list and filename list, work out the total number of files from the
            # length of the file list.
            library_files = images
            num_of_files = len(library_files)
            # Work out the number of rows required by dividing the number of files by the number of
            # columns and rounding up using `math.ceil`.
            num_of_rows_req = ceil(num_of_files / number_of_columns)
            # Create the required number of rows (st.container).
            library_rows = list()
            library_rows_idx = 0
            for i in range(num_of_rows_req):
                library_rows.append(st.container())
            # For each library row we need to create separate rows (st.container) for images,
            # and rows (st.expander) for details and buttons to keep them in the correct columns.
            for idx in range(num_of_rows_req):
                with library_rows[library_rows_idx]:
                    imgs_columns = list(st.columns(number_of_columns))
                # Since we are keeping track of the column and filename indexes we can use
                # those to slice the `library_files` list at the correct points for each row
                # and then increase or reset the indexes as required.
                for img in library_files[filename_idx:(filename_idx + number_of_columns)]:
                    with imgs_columns[col_idx]:
                        st.image(img, use_column_width='auto')
                        st.write(
                            f"""<style>
                                [data-testid="stHorizontalBlock"] {{
                                    align-items: {image_alignment};
                                }}
                                </style>
                                """,
                            unsafe_allow_html=True
                        )
                    # Keeps track of the current column, if we reach the `max_idx` we reset it
                    # to 0 and increase the row index. This combined with the slicing should
                    # ensure all images, details and buttons are in the correct columns.
                    if col_idx < max_idx:
                        col_idx += 1
                    else:
                        col_idx = 0
                        library_rows_idx += 1
                    filename_idx += 1
        return root_container
