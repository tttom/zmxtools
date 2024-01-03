#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to format strings, or objects that can be converted to strings, as unicode subscript or superscript.

Example:
::
    import utils.script
    test_text = '[* 0.123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ +-*=() βɣρΦϕɸχ ]'
    print(f'    regular: {test_text}')
    print(f'  subscript: {script.sub(test_text)}')
    print(f'superscript: {script.super(test_text)}')

"""

__all__ = ['sub', 'super']

__SUB = str.maketrans('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()βɣρΦϕɸχ.' + '*',
                      '₀₁₂₃₄₅₆₇₈₉ₐbcdₑfgₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥwₓyzABCDₑFGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓYZ₊₋₌₍₎ᵦᵧᵨᵩᵩᵩᵪٜ' + '͙'  # '͙'  # or '⁎'
                      )
__SUP = str.maketrans('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()βɣρΦϕɸχ.' + '*',
                      '⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᑫᴿˢᵀᵁᵛᵂˣʸᶻ⁺⁻⁼⁽⁾ᵝᵞ۹ᵠᵠᵠˣॱ' + '⃰'
                      )


def sub(text) -> str:
    """
    Formats the input as a subscript unicode string.
    Note that several letters are missing in unicode and are thus substituted by their case variants or regular letters.

    :param text: A regular txt string or anything that implements the __str__ method.
    :return: The subscript string.
    """
    return str(text).translate(__SUB)


def super(text) -> str:
    """
    Formats the input as a superscript unicode string.
    Note that the lowercase 'q' and several uppercase letters are missing in unicode.
    These are substituted by their case variants or regular letters.

    :param text: A regular txt string or anything that implements the __str__ method.
    :return: The superscript string.
    """
    return str(text).translate(__SUP)
