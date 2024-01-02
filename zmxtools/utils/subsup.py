#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to format strings, or objects that can be converted to strings, as unicode subscript or superscript.
"""

__all__ = ['subscript', 'superscript']

__SUB = str.maketrans('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()βɣρΦϕɸχ.' + '*',
                      '₀₁₂₃₄₅₆₇₈₉ₐbcdₑfgₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥwₓyzABCDₑFGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓYZ₊₋₌₍₎ᵦᵧᵨᵩᵩᵩᵪٜ' + '͙'  # '͙'  # or '⁎'
                      )
__SUP = str.maketrans('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()βɣρΦϕɸχ.' + '*',
                      '⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᑫᴿˢᵀᵁᵛᵂˣʸᶻ⁺⁻⁼⁽⁾ᵝᵞ۹ᵠᵠᵠˣॱ' + '⃰'
                      )


def subscript(text) -> str:
    """
    Formats the input as a subscript unicode string.
    Note that several letters are missing in unicode and are thus substituted by their case variants or regular letters.

    :param text: A regular txt string or anything that implements the __str__ method.
    :return: The subscript string.
    """
    return str(text).translate(__SUB)


def superscript(text) -> str:
    """
    Formats the input as a superscript unicode string.
    Note that the lowercase 'q' and several uppercase letters are missing in unicode.
    These are substituted by their case variants or regular letters.

    :param text: A regular txt string or anything that implements the __str__ method.
    :return: The superscript string.
    """
    return str(text).translate(__SUP)


if __name__ == '__main__':
    from optics.experimental import log
    log.getChild(__name__)

    test_text = '[* 0.123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ +-*=() βɣρΦϕɸχ ]'
    log.info(f'    regular: {test_text}')
    log.info(f'  subscript: {subscript(test_text)}')
    log.info(f'superscript: {superscript(test_text)}')

