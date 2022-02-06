CREATE TABLE T1(IDSERIAL NOT NULL PRIMARY KEY,BCOLORVARCHAR,
FCOLORVARCHAR ); INSERT
INTO T1(BCOLOR,FCOLOR)
VALUES
('red','red')
,('red','red')
,('red',NULL)
,(NULL,'red')
,('red','green')
,('red','blue')
,('green','red')
,('green','blue')
,('green','green')
,('blue','red')
,('blue','green')
,('blue','blue')
;

select * from t1;

select
	distinct bcolor
from
	t1
order by bcolor
;

select
	distinct bcolor,
			fcolor
from
	t1
order by
	bcolor,
	fcolor 	
;

select 
	distinct on(bcolor)
			bcolor, fcolor
from 
	t1
order by
	bcolor, fcolor 
;

select
	distinct on(bcolor)
	bcolor, fcolor
from
	t1 t 
order by
	bcolor, fcolor desc;
