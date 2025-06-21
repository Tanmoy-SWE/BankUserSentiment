# test.py - Test if all packages are installed correctly
import streamlit as st

st.write("Testing if Streamlit works!")

try:
    import pandas as pd
        st.success("✅ Pandas imported successfully")
        except:
            st.error("❌ Pandas import failed")

            try:
                import plotly
                    st.success("✅ Plotly imported successfully")
                    except:
                        st.error("❌ Plotly import failed")

                        try:
                            from textblob import TextBlob
                                st.success("✅ TextBlob imported successfully")
                                except:
                                    st.error("❌ TextBlob import failed")

                                    try:
                                        import nltk
                                            st.success("✅ NLTK imported successfully")
                                            except:
                                                st.error("❌ NLTK import failed")

                                                try:
                                                    import openai
                                                        st.success("✅ OpenAI imported successfully")
                                                        except:
                                                            st.error("❌ OpenAI import failed")