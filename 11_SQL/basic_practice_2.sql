select 
	a.customer_id,
	a.email
from
	customer a
where 
	a.email not like '@%' 
	and 
	a.email not like '%@'
	and 
	a.email like '%@%'
order by a.customer_id;



-- º¹±Í
select 
	c.customer_id,
	c.email
from
	customer c
where 
	c.email not like '%@'
	and 
	c.email not like '@%'
	and 
	c.email like '%@%';
	

select 
	top 3 c.customer_id,
	c.email
from
	customer c
where 
	c.email not like '%@'
	and 
	c.email not like '@%'
	and 
	c.email like '%@%';