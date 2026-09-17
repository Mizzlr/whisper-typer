"""Coldark-inspired token colors with bounded syntax work for raw source views."""
from pygments.style import Style
from pygments.token import Text, Comment, Keyword, Name, String, Number, Operator, Punctuation, Generic, Error

# Palette reference: https://github.com/PrismJS/prism-themes (Coldark).
class ColdarkCold(Style):
    background_color = '#ffffff'
    styles = {Text:'#111b27', Comment:'#3c526d', Keyword:'#a04900', Operator:'#a04900',
              Punctuation:'#111b27', Name:'#111b27', Name.Tag:'#006d6d', Name.Attribute:'#755f00',
              Name.Variable:'#005a8e', Name.Class:'#005a8e', Name.Function:'#7c00aa', Name.Builtin:'#af00af',
              String:'#116b00', String.Regex:'#af00af', Number:'#755f00',
              Generic.Deleted:'#c22f2e', Generic.Inserted:'#116b00', Error:'#c22f2e'}

class ColdarkDark(Style):
    background_color = '#000000'
    styles = {Text:'#e3eaf2', Comment:'#8da1b9', Keyword:'#e9ae7e', Operator:'#e9ae7e',
              Punctuation:'#e3eaf2', Name:'#e3eaf2', Name.Tag:'#66cccc', Name.Attribute:'#e6d37a',
              Name.Variable:'#6cb8e6', Name.Class:'#6cb8e6', Name.Function:'#c699e3', Name.Builtin:'#f4adf4',
              String:'#91d076', String.Regex:'#f4adf4', Number:'#e6d37a',
              Generic.Deleted:'#cd6660', Generic.Inserted:'#91d076', Error:'#cd6660'}

MAX_SYNTAX_CHARS = 150_000
MAX_SYNTAX_LINE = 2_000


def syntax_safe(text):
    # Reject before creating a lexer or tokenizing; retain the complete raw text.
    if len(text) > MAX_SYNTAX_CHARS:
        return False
    start = 0
    for end, char in enumerate(text):
        if char in '\r\n':
            start = end + 1
        elif end - start >= MAX_SYNTAX_LINE:
            return False
    return True


def source_style(dark):
    return ColdarkDark if dark else ColdarkCold
