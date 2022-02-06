-- 문제 7

select * from address limit 10;

select
	address_id,
	address,
	district,
	postal_code,
	substring(postal_code,2,1) as sub_2
from
	address
where 
	postal_code like '_1%';
	

select 
	address_id,
	address,
	district,
	postal_code,
	substring(postal_code,2,1) test1
from
	address a 
where 
	substring(postal_code,2,1)= '1';
	

-- 문제 12번
select * from address limit 10;
select 
	*
from 
	address a 
where 
	postal_code = ''
or	postal_code = '35200'
or 	postal_code = '17886';


select
	*,
	case when postal_code = '' then 'empty'
		 else postal_code
	end as postal_code_emptyflag
from
	address a 
where 
	postal_code = ''
or	postal_code = '35200'
or 	postal_code = '17886';	


-- 문제 16
select * from address limit 10;

  -- 정담 1
select 
	*
from 
	address a 
where 
	address2 is null 
or	postal_code = '35200'
or	postal_code = '17886';

  -- 정답 2
select 
	*
from 
	address a 
where 
	address2 is null 
or	postal_code in ('35200', '17886');




select 
	*
from 
	address a 
where 
	address2 is null 
order by
	address2 desc;

select 
	*
from 
	address a 
where 
	address2 = '' 
order by
	address2 desc;
	

-- 문제 17
select * from customer limit 10;
select 
	*
from 
	customer c 
where 
	last_name like '%Jone%';