from __future__ import annotations

import os

import streamlit as st

import app_dev
import app_user


def get_dev_password() -> str:
    env_password = os.getenv("DEV_PASSWORD")
    if env_password:
        return env_password
    try:
        return st.secrets.get("DEV_PASSWORD", "admin")
    except Exception:
        return "admin"


def init_session_state() -> None:
    st.session_state.setdefault("current_mode", "home")
    st.session_state.setdefault("is_authenticated", False)


def go(mode: str) -> None:
    st.session_state.current_mode = mode
    st.rerun()


st.set_page_config(page_title="AI 智慧捷運路徑規劃", layout="wide")
init_session_state()


if st.session_state.current_mode == "home":
    st.title("AI 智慧捷運路徑規劃系統")
    st.write("請選擇進入前台使用者模式，或進入開發者後台檢查資料。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("前台使用者模式", use_container_width=True, type="primary"):
            go("user")
    with col2:
        if st.button("開發者後台", use_container_width=True):
            go("dev_login")

elif st.session_state.current_mode == "user":
    app_user.run()
    st.sidebar.divider()
    if st.sidebar.button("返回入口"):
        go("home")

elif st.session_state.current_mode == "dev_login":
    st.title("開發者權限驗證")
    pwd_input = st.text_input("請輸入密碼", type="password")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("登入", type="primary"):
            if pwd_input == get_dev_password():
                st.session_state.is_authenticated = True
                go("dev_dashboard")
            else:
                st.error("密碼錯誤。")
    with col2:
        if st.button("取消"):
            go("home")

elif st.session_state.current_mode == "dev_dashboard" and st.session_state.is_authenticated:
    app_dev.run()
    st.sidebar.divider()
    if st.sidebar.button("登出"):
        st.session_state.is_authenticated = False
        go("home")

else:
    st.session_state.is_authenticated = False
    go("home")
