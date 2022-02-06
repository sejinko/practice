select 
	*
from 
	sales2007_1 s
union all
select
	*
from 
	sales2007_2 d;


select 
	name
from
	sales2007_1
union all
select
	name
from
	sales2007_2;


select 
	amount
from
	sales2007_1 s 
union all
select
	amount 
from
	sales2007_2 s2 ;
	

select 
	*
from 
	sales2007_1
union all
select
	*
from 
	sales2007_2;
 