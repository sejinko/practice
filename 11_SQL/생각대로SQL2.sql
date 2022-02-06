-- 문제 1번
select * from film;
select 
	*
from 
	film
limit 100;


-- 문제 2번
select * from actor;
select 
	actor_id,
	last_name,
	first_name
from
	actor
where 
	last_name like 'Jo%'  -- 대문자, 소문자 구별함
order by
	actor_id
limit 1;


-- 문제 3번
select * from country limit 10;
select 
	*
from 
	country
where 
	country_id between 2 and 9;


-- 문제 4번
select * from country limit 10;
select 
	country
from
	country
where 
	country like 'A%';


-- 문제 5번
select 
	country
from
	country
where
	country like '%s';


-- 문제 6번
select * from address limit 10;
select 
	address_id,
	address,
	district,
	postal_code
from
	address
where
	postal_code like '77%';


-- 문제 7번
select
	address_id,
	address,
	district,
	postal_code
from
	address
where 
	postal_code like '_1%';


-- 문제 8번
select * from payment limit 10;
select 
	*
from 
	payment
where 
	customer_id = 341
and to_char(payment_date, 'yyyymmdd') between '20070215' and '20070216';


-- 문제 9번
  -- 정답 1
select
	*
from 
	payment
where 
	customer_id = 355
and amount between 1 and 3;
  -- 정답 2
select 
	*
from 
	payment
where 
	customer_id = 355
and amount > 1
and amount < 3;


-- 문제 10번
select * from customer limit 10;

select 
	customer_id,
	last_name,
	first_name
from 
	customer c 
where 
	first_name in ('Maria', 'Lisa', 'Mike');


-- 문제 11번
select * from film limit 10;
select
	*
from
	film f 
where 
	(100 <= length and length <= 120)
or  (3 <= rental_duration and rental_duration <= 5);


select
	*
from
	film f 
where 
	length between 100 and 120
or  rental_duration between 3 and 5;


	
-- 문제 12번
select * from address limit 10 ;
select 
	*
from 
	address a 
where 
	postal_code in ('', '45200', '17886');


-- 문제 13번
select * from address limit 10;
select 
	*
from 
	address a 
where 
	address2 isnull;  -- isnull 그냥 이렇게 쓰면 되나?
	
	
-- 문제 14번
select * from staff limit 10;
select 
	staff_id,
	first_name || ' ' || last_name as name
from 
	staff s 
where 
	picture notnull;  -- notnull 이렇게 쓰면 되나?
	
	
-- 문제 15번
select * from rental limit 10;
select 
	*
from 
	rental r 
where
	return_date isnull;

select 
	count(*)
from 
	rental r 
where
	return_date isnull;


-- 문제 16번
select * from address limit 10;
select 
	*
from 
	address a 
where 
	postal_code in ('', '35200', '17886');  -- null이면 빈값으로 들어가나? null이라고 적혀있는 값은 없는가?

	
-- 문제 17번
select * from customer limit 10;
select
	*
from 
	customer c 
where 
	last_name like '%John%';  -- %는 있어도 되고 없어도 되는 구나


-- 문제 18번
select * from address limit 10;
select
	*
from 
	address a
where 
	address2 is null;


