# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:34:00 2023

@author: KITCOOP
"""

from django import template

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter
def nextpage(pageNum, pageview):
    page = (pageNum+pageview-1) // pageview * pageview + 1 # pageNum + pageview -1
    return page

@register.filter
def prepage(pageNum, pageview):
    page = (pageNum - pageview - 1) // pageview * pageview + 1 #pageNum - pageview -1
    return page



# 번호 = 전체건수(board.paginator.count) - 시작인덱스(board.start_index)
# - 현재인덱스(forloop.counter0)