-- SQL1

-- 문제 7번
select * from rental limit 10;
	
SELECT 
	*
	--date(r.rental_date),
	--julianday(date(return_date)) as origin_date,
	--lag(julianday(date(return_date))) over() as lag_date,
	--floor(julianday(date(return_date)) - julianday(date(rental_date))) + 1 as diff_date
FROM 
	rental r
WHERE 
	floor(julianday(date(return_date)) - julianday(date(rental_date))) + 1 > 10;


-- 문제 8번
select * from customer limit 10;

SELECT 
	customer_id,
	first_name,
	last_name
from 
	customer c
WHERE 
	customer_id/50 = 0;

-- 문제 10번
select * from city limit 10;

SELECT 
	count(distinct(city))
from 
	city;

SELECT 
	count(city_id)
from 
	city;


-- 문제 11번
select * from actor limit 10;

SELECT 
	upper(first_name || ' ' || last_name) as upper_name
	--upper(concat(concat(first_name, ' '), last_name)) -- sqlite에서는 안됨
from 
	actor;



-- SQL2

-- 문제 7번
select * from address limit 10;
  -- 1번째 정답
SELECT 
	address_id,
	address,
	district,
	postal_code
from 
	address a 
WHERE 
	postal_code like '_1%';
	
-- 2번째 정답
SELECT 
	address_id,
	address,
	district,
	postal_code,
	substring(postal_code,2,1)
from 
	address a 
WHERE 
	substring(postal_code,2,1) = '1';


-- 문제 12번
select * from address limit 10;

SELECT 
	*,
	case when postal_code = '' then 'empty'
		 when postal_code = 35200 then '35200'
		 when postal_code = 17886 then '17886'
	end as postal_code_2
FROM 
	address a 
WHERE 
	postal_code in ('', 35200, 17886);
	

-- 문제 16번
select * from address limit 10;

SELECT 
	*
from 
	address a 
WHERE 
	address2 is null
or	postal_code = 35200
or	postal_code = 17886;


-- 문제 17번
select * from customer limit 10;

  -- 1번 정답
SELECT 
	first_name,
	last_name
FROM 
	customer c 
WHERE 
	last_name like '%John%';
	
  -- 2번 정답
SELECT 
	first_name,
	last_name
FROM 
	customer c 
WHERE 
	lower(last_name) like '%john%';


-- SQL3

-- 문제 2번
select * from customer limit 10;
select * from address limit 10;
select * from city limit 10;

SELECT 
	c.customer_id,
	c.first_name,
	c.last_name,
	a.address,
	a.district,
	a.postal_code,
	a.phone,
	c2.city
from 
	customer c
	join address a on  c.address_id = a.address_id 
	join city c2 on c2.city_id = a.city_id;


-- 문제 3번
select * from address;
select * from customer;
select * from city;


SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone
from 
	address a
	join customer c on a.address_id = c.address_id
	join city c2 on a.city_id = c2.city_id 
WHERE 
	city like '%Lima%';

SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone
from 
	address a
	join customer c on a.address_id = c.address_id
	join city c2 on a.city_id = c2.city_id 
WHERE 
	city = 'Lima';

SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone
from 
	address a
	join customer c on a.address_id = c.address_id
	join city c2 on a.city_id = c2.city_id 
WHERE 
	city in ('Lima');


-- 문제 10번
select * from country limit 10;
select * from customer limit 10;
select * from city limit 10;
select * from address limit 10;

SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone,
	c2.country,
	c3.city 
from 
	customer c
	join address a on c.address_id = a.address_id 
	join city c3 on c3.city_id = c.customer_id 
	join country c2 on c3.country_id  = c2.country_id 
WHERE 
	lower(c2.country) != 'china';

SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone,
	c2.country,
	c3.city 
from 
	customer c
	join address a on c.address_id = a.address_id 
	join city c3 on c3.city_id = c.customer_id 
	join country c2 on c3.country_id  = c2.country_id 
WHERE 
	lower(c2.country) not in ('china');	
	
SELECT 
	c.first_name,
	c.last_name,
	c.email,
	a.phone,
	c2.country,
	c3.city 
from 
	customer c
	join address a on c.address_id = a.address_id 
	join city c3 on c3.city_id = c.customer_id 
	join country c2 on c3.country_id  = c2.country_id 
WHERE 
	lower(c2.country) like '%china%';
	

-- 문제 13
SELECT * 
	
	