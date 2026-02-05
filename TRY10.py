streamlit.errors.StreamlitInvalidWidthError: Invalid width value: 'container'. Width must be either a positive integer (pixels), 'stretch', or 'content'.

Traceback:
File "/mount/src/skin-question/TRY10.py", line 471, in <module>
    main()
    ~~~~^^
File "/mount/src/skin-question/TRY10.py", line 468, in main
    result_step()
    ~~~~~~~~~~~^^
File "/mount/src/skin-question/TRY10.py", line 424, in result_step
    st.bar_chart(acc_data, color="#3498db", width="container")
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 532, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/vega_charts.py", line 1524, in bar_chart
    self._altair_chart(
    ~~~~~~~~~~~~~~~~~~^
        chart,
        ^^^^^^
    ...<4 lines>...
        height=height,
        ^^^^^^^^^^^^^^
    ),
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/vega_charts.py", line 2274, in _altair_chart
    return self._vega_lite_chart(
           ~~~~~~~~~~~~~~~~~~~~~^
        data=None,  # The data is already part of the spec
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<8 lines>...
        height=height,
        ^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/vega_charts.py", line 2390, in _vega_lite_chart
    validate_width(width, allow_content=True)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/layout_utils.py", line 96, in validate_width
    raise StreamlitInvalidWidthError(width, allow_content)
