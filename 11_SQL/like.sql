select 
	first_name,
	last_name
from 
	customer
where 
	first_name like 'Jen%';
	

select 
	first_name,
	last_name
from
	customer
where
	first_name like '%er%';
	

select
	first_name,
	last_name
from
	customer
where 
	first_name not like 'len%';
	

