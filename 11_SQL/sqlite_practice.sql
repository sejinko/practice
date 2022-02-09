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
SELECT * from actor limit 10;
select * from film_actor limit 10;
select * from film limit 10;

SELECT 
	a.actor_id,
	a.first_name,
	a.last_name,
	angels_actor.actor_id,
	case
		when angels_actor.title = 'Angels Life' then 'Y'
		else 'N'
	end as 'angelslife_flag'
from 
	actor a 
	left outer join
	(
		select
			f.film_id,
			f.title,
			fa.actor_id
		from 
			film f
			join film_actor fa on f.film_id = fa.film_id 
		WHERE 
			f.title = 'Angels Life'
	) as angels_actor
		on a.actor_id = angels_actor.actor_id;
	

-- 문제 14
select * from rental;
select * from customer;
select * from staff;


select
	*	
FROM 
	rental r
WHERE 
	date(rental_date) >= '2005-06-01'
AND date(rental_date) <= '2005-06-14';

select
	r.*,
	c.first_name || ' ' || c.last_name as fullname_customer,
	s.first_name || ' ' || s.last_name as fullname_employee
FROM 
	rental r
	join customer c on r.customer_id  = c.customer_id
	join staff s on r.staff_id = s.staff_id 
WHERE 
	date(rental_date) between '2005-06-01' AND '2005-06-14'
and	(c.first_name || ' ' || c.last_name = 'Gloria Cook'
or	s.first_name || ' ' || s.last_name = 'Mike Hillyer');


-- SQL 4

-- 문제 2
select * from film limit 10;

SELECT 
	f.rating,
	count(film_id)
from 
	film f
group by
	f.rating;

-- 문제 4
select * from actor;
select * from film_actor;

SELECT 
	fa.actor_id,
	a.first_name || ' ' || a.last_name as full_name,
	count(distinct film_id)
from 
	actor a
	join film_actor fa on a.actor_id = fa.actor_id
group by
	fa.actor_id
	
  -- 검증
SELECT 
	d.*,
	a.first_name,
	a.last_name 
from 
	(
		select 
			actor_id,
			count(distinct film_id) as cnt
		FROM 
			film_actor as fa 
		group by 
			actor_id 
	) as d
	left outer join actor a on d.actor_id = a.actor_id ;

-- 8번

  -- 첫번째
select * from rental limit 10;

SELECT 
	customer_id,
	count(distinct rental_id)
FROM 
	rental r
where
	date(r.rental_date) = '2005-05-26'
group by 
	customer_id
HAVING 
	count(distinct r.rental_id) >= 2;
  -- 두번째
select * from rental limit 10;

SELECT 
	customer_id,
	count(distinct rental_id)
FROM 
	rental r
where
	r.rental_date between '2005-05-26 00:00:00' and '2005-05-26 23:59:59'
group by 
	customer_id
HAVING 
	count(distinct r.rental_id) >= 2;
	
-- 13번
select * from payment limit 10;
select * from rental limit 10;

  -- 1번
SELECT 
	cust_rating,
	count(customer_id) as cnt
from 
	(
		select 
			db.customer_id ,
			db.sum_amount,
			case
				when db.sum_amount >= 151 then 'A'
				when db.sum_amount between 101 and 150 then 'B'
				when db.sum_amount between 51 and 100 then 'C'
				when db.sum_amount <= 50 then 'D'
				else 'Empty'
			end as cust_rating
		FROM 
			(
				select 
					p.customer_id,
					round(sum(p.amount),0) as sum_amount
				from 	
					payment p
					join rental r on p.rental_id = r.rental_id
						and	r.customer_id = p.customer_id 
						and r.staff_id = p.staff_id 
				group by 
					r.customer_id
			) as db
	) as db
group by 
	cust_rating;

  -- 두번째
SELECT 
	db.rating,
	count(db.rating)
from
	(
		SELECT 
			r.customer_id ,
			round(sum(p.amount), 0) as total_amount,
			case
				when sum(p.amount) >= 151 then 'A'
				when sum(p.amount) between 101 and 150 then 'B'
				when sum(p.amount) between 51 and 100 then 'C'
				when sum(p.amount) <= 50 then 'D'
				else 'Empty'
			end as rating
		from 
			payment p
			join rental r on p.rental_id = r.rental_id
				and	r.customer_id = p.customer_id 
				and r.staff_id = p.staff_id 
		group by 
			r.customer_id
	) as db
group by 
	db.rating;
  -- 답
SELECT 
	db.rating,
	count(db.rating)
from 
	(
		SELECT 
		r.customer_id ,
		round(p.amount, 1) as total_amount,
		case
			when round(p.amount, 1) >= 150 then 'A'
			when round(p.amount, 1) between 101 and 150 then 'B'
			when round(p.amount, 1) between 51 and 100 then 'C'
			when round(p.amount, 1) <= 50 then 'D'
			end as rating
		from 
			payment p
			join rental r on p.rental_id = r.rental_id
		group by 
			r.customer_id
	) as db 
group BY 
	db.rating;


-- SQL 5

 -- 1번
SELECT * from film limit 10;
select * from actor limit 10;
select * from film_actor limit 10;

SELECT 
	actor_id,
	'over_length180' as flag
FROM 
	film_actor fa
WHERE 
	film_id in
	(
		select
			film_id 
		from 
			film
		WHERE 
			length >= 180
	)
union
select 
	actor_id,
	'rating_R' as flag
FROM 
	film_actor fa
WHERE 
	film_id in
	(
		select
			film_id 
		from 
			film
		WHERE 
			length >= 180
	);


-- 4번
SELECT
	film_id 
from 
	film film 
except
select 
	f.film_id 
from 
	film f
	join film_category fc on f.film_id = fc.film_id 
	join category c on fc.category_id = c.category_id 
WHERE 
	c.name in ('Action', 'Animation', 'Horror');

-- 6번
select * from customer limit 10;
select * from employee limit 10;

SELECT
	first_name || ' ' || last_name as fullname
from 
	employee c 
except
select 
	first_name || ' ' || last_name as fullname
from
	customer;

SELECT
	first_name || ' ' || last_name as fullname
from 
	employee c ;


-- 8번
select * from payment limit 10;
select * from country limit 10;
select * from customer limit 10;
select * from address limit 10;
select * from city limit 10;

SELECT 
	country,
	city,
	sum(amount)
FROM 
	payment p
	join customer c2 on c2.customer_id = p.customer_id 
	join address a on a.address_id = c2.address_id 
	join city c3 on c3.city_id = a.city_id 
	join country c on c.country_id  = c3.country_id 
group BY 
	country, city
union all
SELECT 
	country,
	null,
	sum(amount)
FROM 
	payment p
	join customer c2 on c2.customer_id = p.customer_id 
	join address a on a.address_id = c2.address_id 
	join city c3 on c3.city_id = a.city_id 
	join country c on c.country_id  = c3.country_id 
group BY 
	country
union all
SELECT 
	null,
	null,
	sum(amount)
FROM 
	payment p
	join customer c2 on c2.customer_id = p.customer_id 
	join address a on a.address_id = c2.address_id 
	join city c3 on c3.city_id = a.city_id 
	join country c on c.country_id  = c3.country_id;
	
	
-- SQL 6


-- 1번
select * from payment limit 10;

  -- 첫번째
SELECT 
	customer_id,
	fullname,
	sum_amount
from
	(
		select 
			c.customer_id,
			c.first_name || ' ' || c.last_name as fullname,
			sum(p.amount) as sum_amount
		from 
			payment p
			join customer c on c.customer_id = p.customer_id 
		group by
			c.customer_id
	) as db
order BY 
	sum_amount desc
LIMIT 
	1;
	
  -- 두번째
SELECT 
	first_name,
	last_name 
from 
	customer c 
WHERE 
	customer_id in
	(
		SELECT 
			customer_id 
		from 
			payment p
		group by 
			customer_id 
		order by
			sum(amount) DESC 
		limit 1
	)

	
-- 2번
SELECT * from category limit 10;
select * from rental limit 10;
select * from film limit 10;
select * from customer limit 10;
select * from film_category limit 10;
select * from inventory limit 10;

SELECT 
	*
from 
	category c 
WHERE 
	EXISTS 
	(
		SELECT 
			1
		from 
			rental r
			join inventory i2 on r.inventory_id = i2.inventory_id 
			join film_category fc on i2.film_id = fc.film_id 
		WHERE 
			c.category_id = fc.category_id 
	)

	
-- 3번
-- sqlite에는 any가 없음
SELECT 
	*
FROM 
	category c 
WHERE
	category_id = any 
	(
		select
			fc.category_id 
		from 
			rental r 
			join inventory i2 on r.inventory_id = i2.inventory_id 
			join film_category fc on i2.film_id = fc.film_id
		where fc.category_id in (1,2)
	);


-- 5번
SELECT * from rental limit 10;

SELECT 
	r.customer_id,
	r.rental_id 
from 
	rental r
	join customer c on r.customer_id = c.customer_id
WHERE r.customer_id in
	(
		SELECT 
			customer_id 
		from 
			rental
		group by 
			customer_id 
		order by
			count(distinct rental_id) DESC 
		LIMIT 
		 	1
	);


SELECT 
	first_name,
	last_name
from 
	customer c 
where customer_id in
	(
		SELECT 
			customer_id
		from 
			rental r 
		group by 
			customer_id
		order by
			count(rental_id) desc
		LIMIT 1
	)


-- 6번
select * from film_category limit 10;	

SELECT 
	*
FROM 
	film f 
where film_id not in
	(
		SELECT 
			film_id 
		from 
			film_category fc
	);

	
SELECT 
	*
FROM 
	film f
where not EXISTS 
	(
		SELECT 
			1
		from 
			film_category fc 
		WHERE 
			fc.film_id = f.film_id 
	)
	
	
	
select
	*
FROM 
	address a
WHERE 
	not exists
	(
		SELECT 
			1
		from 
			(
				select 
					'' as a
			) as db
		where db.a = a.address2
	);

address2 != ''

SELECT 
	*
from address a 
where address2 not in
	(
		select ''
	);


SELECT 
	*
from
	address a 
where
	address2 not null
	
	
	
-- SQL 7
	
-- 3번
-- sqlite에서는 grouping sets가 안됨
	
SELECT 
	c3.country,
	c2.city,
	sum(p.amount) as sum_amount
FROM 
	payment p
	join customer c on p.customer_id = c.customer_id 
	join address a on c.address_id = a.address_id
	join city c2 on a.city_id = c2.city_id
	join country c3 on c2.country_id = c3.country_id 
group BY 
grouping sets 
	((c3.country, c2.city), (c3.country), ());
	
-- 4번
-- sqlite에서는 rollup 함수도 없음
SELECT 
	c3.country_id,
	c2.city_id,
	sum(p.amount) as sum_amount
FROM 
	payment p
	join customer c on p.customer_id = c.customer_id 
	join address a on c.address_id = a.address_id
	join city c2 on a.city_id = c2.city_id
	join country c3 on c2.country_id = c3.country_id 
group BY 
rollup (c3.country, c2.city);

-- 5번
SELECT * from actor limit 10;
select * from film_actor limit 10;

SELECT 
	actor_id,
	count(distinct film_id) as film_count,
	sum(count(distinct film_id)) over()
from 
	film_actor
group by
	actor_id;

   -- 합계	
select
	sum(film_count)
FROM 
	(
		SELECT 
			actor_id,
			count(distinct film_id) as film_count
		from 
			film_actor
		group by
			actor_id
			
-- 7번
	);